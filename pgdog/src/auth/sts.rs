//! AWS STS client authentication: presigned GetCallerIdentity URL validation.
//!
//! Clients present a presigned STS `GetCallerIdentity` URL as their Postgres
//! password (the aws-iam-authenticator pattern). [`verify`] runs the whole
//! handshake: syntactic validation of the presigned URL ([`precheck`]), a
//! `GetCallerIdentity` round-trip to STS, IAM role ARN normalization and
//! matching against the user's `allowed_iam_arns`, and a cache of
//! already-verified tokens.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use aws_lc_rs::digest;
use chrono::NaiveDateTime;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use reqwest::StatusCode;
use reqwest::header::ACCEPT;
use reqwest::redirect;
use serde::Deserialize;
use thiserror::Error;
use tokio::sync::{Semaphore, SemaphorePermit};
use tracing::warn;
use url::Url;

/// Header clients must include in `X-Amz-SignedHeaders`, carrying the
/// audience value from `sts_server_id` in `pgdog.toml`.
pub const SERVER_ID_HEADER: &str = "x-pgdog-server-id";

/// Maximum allowed value of `X-Amz-Expires`, in seconds.
const MAX_EXPIRES: u64 = 900;

/// Upper bound on how long a verified token stays cached after insertion.
const CACHE_MAX_TTL: Duration = Duration::from_secs(300);

/// Total timeout for the outbound `GetCallerIdentity` request to STS.
const STS_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a failed verification (STS rejection, transport failure,
/// unparsable response, disallowed ARN) is remembered.
///
/// Precheck only validates syntax, so without this an attacker with no AWS
/// credentials could mint well-formed presigned URLs (garbage signature) and
/// turn every connection attempt into a fresh outbound STS call — and STS
/// throttles per *account*, so sustained abuse could starve legitimate
/// authentication. Kept short so a client retrying with fixed credentials
/// recovers quickly.
const NEGATIVE_CACHE_TTL: Duration = Duration::from_secs(30);

/// Upper bound on the STS response body size. Real `GetCallerIdentity`
/// responses are well under 1KB; anything bigger isn't worth parsing.
const MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// Maximum concurrent outbound STS verifications.
///
/// The negative cache only defuses repeated *identical* tokens; an
/// unauthenticated client minting distinct well-formed presigned URLs
/// (garbage signatures) would otherwise get one outbound round-trip per
/// token, each holding a socket for up to [`STS_REQUEST_TIMEOUT`] —
/// and STS throttles per account. Excess attempts are rejected
/// immediately rather than queued: a queue would just move the
/// resource exhaustion.
const MAX_INFLIGHT_VERIFICATIONS: usize = 16;

static INFLIGHT: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(MAX_INFLIGHT_VERIFICATIONS));

/// Reserve one of the [`MAX_INFLIGHT_VERIFICATIONS`] slots, failing
/// immediately when all are taken. The permit is a RAII guard: it's
/// released when dropped, on error and panic paths included.
fn inflight_permit() -> Result<SemaphorePermit<'static>, StsAuthError> {
    INFLIGHT
        .try_acquire()
        .map_err(|_| StsAuthError::TooManyVerifications)
}

/// Reasons a presigned STS URL fails validation before any I/O happens.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum StsAuthError {
    /// The password doesn't parse as a URL.
    #[error("password is not a valid URL")]
    InvalidUrl,

    /// The URL scheme is not `https`.
    #[error("URL scheme must be \"https\"")]
    InvalidScheme,

    /// The host is not a recognized STS endpoint.
    #[error("host is not an STS endpoint")]
    InvalidHost,

    /// The URL specifies a port other than the https default (443).
    #[error("URL must use the default https port")]
    InvalidPort,

    /// The URL path is not `/`.
    #[error("URL path must be \"/\"")]
    InvalidPath,

    /// The URL is not a `GetCallerIdentity` request.
    #[error("\"Action\" must be \"GetCallerIdentity\"")]
    InvalidAction,

    /// The signing algorithm is not SigV4.
    #[error("\"X-Amz-Algorithm\" must be \"AWS4-HMAC-SHA256\"")]
    InvalidAlgorithm,

    /// `X-Amz-Expires` is missing or not an integer.
    #[error("\"X-Amz-Expires\" is missing or not an integer")]
    InvalidExpires,

    /// `X-Amz-Expires` exceeds the maximum allowed lifetime.
    #[error("\"X-Amz-Expires\" exceeds {MAX_EXPIRES} seconds")]
    ExpiresTooLong,

    /// `X-Amz-Date` is missing or not in `YYYYMMDDTHHMMSSZ` format.
    #[error("\"X-Amz-Date\" is missing or malformed")]
    InvalidDate,

    /// The presigned URL has already expired.
    #[error("presigned URL has expired")]
    Expired,

    /// `X-Amz-SignedHeaders` is missing from the query string.
    #[error("\"X-Amz-SignedHeaders\" is missing")]
    MissingSignedHeaders,

    /// `X-Amz-SignedHeaders` doesn't include `host`.
    #[error("\"X-Amz-SignedHeaders\" does not include \"host\"")]
    HostHeaderNotSigned,

    /// `X-Amz-SignedHeaders` doesn't include the pgdog audience header.
    #[error("\"X-Amz-SignedHeaders\" does not include \"{SERVER_ID_HEADER}\"")]
    ServerIdHeaderNotSigned,

    /// The verification request to STS couldn't be completed at all
    /// (connect error, timeout, invalid response transport).
    #[error("request to STS failed: {0}")]
    RequestFailed(String),

    /// STS returned a non-200 response: the signature (audience header
    /// included) didn't validate, or the credentials expired or were revoked.
    #[error("STS rejected the token (HTTP status {0})")]
    StsRejected(u16),

    /// The STS response body isn't a valid JSON `GetCallerIdentityResponse`.
    #[error("STS response is not a valid GetCallerIdentityResponse: {0}")]
    InvalidResponse(String),

    /// The verified caller identity is not in the user's `allowed_iam_arns`.
    #[error("caller identity is not in \"allowed_iam_arns\"")]
    ArnNotAllowed,

    /// Too many STS verification requests are already in flight; the
    /// attempt is rejected immediately instead of being queued.
    #[error("too many concurrent STS verification requests")]
    TooManyVerifications,
}

/// A presigned STS URL that passed offline validation and is ready to be
/// executed against STS.
#[derive(Debug, Clone)]
pub struct PrecheckedToken {
    url: Url,
    server_id: String,
    expires_at: SystemTime,
}

impl PrecheckedToken {
    /// The validated presigned URL.
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// Audience value to send in the [`SERVER_ID_HEADER`] header when
    /// executing the request.
    pub fn server_id(&self) -> &str {
        &self.server_id
    }

    /// When the presigned URL expires (`X-Amz-Date` + `X-Amz-Expires`).
    pub fn expires_at(&self) -> SystemTime {
        self.expires_at
    }
}

/// A token whose presigned URL was executed against STS and mapped to a
/// caller identity.
#[derive(Debug, Clone)]
pub struct ValidatedStsToken {
    /// Caller ARN exactly as returned by STS.
    pub arn: String,
    /// Caller ARN after [`normalize_arn`].
    pub normalized_arn: String,
    /// When the presigned URL expires (`X-Amz-Date` + `X-Amz-Expires`).
    pub expires_at: SystemTime,
    /// Audience (`sts_server_id`) the token was verified against. Cache
    /// hits are only valid for this audience: after `sts_server_id` is
    /// rotated, a token verified under the old value must re-verify.
    pub server_id: String,
}

/// Validate a presigned STS `GetCallerIdentity` URL without performing any
/// I/O.
///
/// `server_id` is this pooler's `sts_server_id`; it's carried into the
/// returned token so the caller can send it in the [`SERVER_ID_HEADER`]
/// header when executing the request.
pub fn precheck(password: &str, server_id: &str) -> Result<PrecheckedToken, StsAuthError> {
    precheck_at(password, server_id, SystemTime::now())
}

fn precheck_at(
    password: &str,
    server_id: &str,
    now: SystemTime,
) -> Result<PrecheckedToken, StsAuthError> {
    let url = Url::parse(password).map_err(|_| StsAuthError::InvalidUrl)?;

    if url.scheme() != "https" {
        return Err(StsAuthError::InvalidScheme);
    }

    if !url.host_str().is_some_and(valid_sts_host) {
        return Err(StsAuthError::InvalidHost);
    }

    // `Url` normalizes an explicit `:443` away, so any remaining port is
    // non-default. STS only listens on 443; an attacker-chosen port would
    // turn the (unauthenticated) verification request into a slow-connect
    // amplification lever.
    if url.port().is_some() {
        return Err(StsAuthError::InvalidPort);
    }

    // GetCallerIdentity is only ever served at the root path.
    if url.path() != "/" {
        return Err(StsAuthError::InvalidPath);
    }

    let mut action = None;
    let mut algorithm = None;
    let mut expires = None;
    let mut date = None;
    let mut signed_headers = None;

    for (name, value) in url.query_pairs() {
        let param = match name.as_ref() {
            "Action" => &mut action,
            "X-Amz-Algorithm" => &mut algorithm,
            "X-Amz-Expires" => &mut expires,
            "X-Amz-Date" => &mut date,
            "X-Amz-SignedHeaders" => &mut signed_headers,
            _ => continue,
        };
        param.get_or_insert(value.into_owned());
    }

    if action.as_deref() != Some("GetCallerIdentity") {
        return Err(StsAuthError::InvalidAction);
    }

    if algorithm.as_deref() != Some("AWS4-HMAC-SHA256") {
        return Err(StsAuthError::InvalidAlgorithm);
    }

    let expires: u64 = expires
        .as_deref()
        .and_then(|e| e.parse().ok())
        .ok_or(StsAuthError::InvalidExpires)?;
    if expires > MAX_EXPIRES {
        return Err(StsAuthError::ExpiresTooLong);
    }

    let date = date
        .as_deref()
        .and_then(parse_amz_date)
        .ok_or(StsAuthError::InvalidDate)?;

    let expires_at = date + Duration::from_secs(expires);
    if expires_at <= now {
        return Err(StsAuthError::Expired);
    }

    let signed_headers = signed_headers.ok_or(StsAuthError::MissingSignedHeaders)?;
    let mut host_signed = false;
    let mut server_id_signed = false;
    for header in signed_headers.split(';') {
        match header.trim().to_ascii_lowercase().as_str() {
            "host" => host_signed = true,
            SERVER_ID_HEADER => server_id_signed = true,
            _ => (),
        }
    }
    if !host_signed {
        return Err(StsAuthError::HostHeaderNotSigned);
    }
    if !server_id_signed {
        return Err(StsAuthError::ServerIdHeaderNotSigned);
    }

    Ok(PrecheckedToken {
        url,
        server_id: server_id.to_owned(),
        expires_at,
    })
}

/// `sts.amazonaws.com` or `sts.<region>.amazonaws.com`
/// where `<region>` is `[a-z0-9-]+`.
fn valid_sts_host(host: &str) -> bool {
    if host == "sts.amazonaws.com" {
        return true;
    }

    let Some(region) = host
        .strip_prefix("sts.")
        .and_then(|rest| rest.strip_suffix(".amazonaws.com"))
    else {
        return false;
    };

    !region.is_empty()
        && region
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Parse a SigV4 `X-Amz-Date` timestamp (`YYYYMMDDTHHMMSSZ`).
fn parse_amz_date(date: &str) -> Option<SystemTime> {
    NaiveDateTime::parse_from_str(date, "%Y%m%dT%H%M%SZ")
        .ok()
        .map(|naive| naive.and_utc().into())
}

/// Map an STS assumed-role ARN to the IAM role ARN it was assumed from:
///
/// `arn:aws:sts::<acct>:assumed-role/<RoleName>/<session>` →
/// `arn:aws:iam::<acct>:role/<RoleName>`
///
/// ARNs that don't match this shape are returned unchanged.
pub fn normalize_arn(arn: &str) -> String {
    let parsed = arn
        .strip_prefix("arn:aws:sts::")
        .and_then(|rest| rest.split_once(':'))
        .and_then(|(account, resource)| {
            let role_session = resource.strip_prefix("assumed-role/")?;
            let (role, session) = role_session.split_once('/')?;
            (!role.is_empty() && !session.is_empty()).then_some((account, role))
        });

    match parsed {
        Some((account, role)) => format!("arn:aws:iam::{}:role/{}", account, role),
        None => arn.to_owned(),
    }
}

/// Whether a caller ARN matches the user's `allowed_iam_arns` list, either
/// by its normalized IAM role ARN or by an exact unnormalized match.
pub fn matches_allowed(normalized: &str, raw: &str, allowed: &[String]) -> bool {
    allowed.iter().any(|arn| arn == normalized || arn == raw)
}

/// Verify a presigned STS token end to end: offline [`precheck`], a
/// `GetCallerIdentity` round-trip to STS, and authorization of the caller
/// identity against the user's `allowed_iam_arns`.
///
/// Verified tokens are cached (keyed by the token alone), so a hit may have
/// been inserted on behalf of a different user: cache hits skip the STS
/// round-trip but are re-authorized against `allowed` every time, and are
/// only served for the audience (`server_id`) they were verified under.
pub async fn verify(
    password: &str,
    server_id: &str,
    allowed: &[String],
    cache: &StsTokenCache,
) -> Result<ValidatedStsToken, StsAuthError> {
    // Positive entries are checked first so a negative entry can never
    // shadow an already-verified identity (hits re-authorize per user).
    if let Some(token) = cache.get(password, server_id) {
        return if matches_allowed(&token.normalized_arn, &token.arn, allowed) {
            Ok(token)
        } else {
            Err(StsAuthError::ArnNotAllowed)
        };
    }

    // Recently-failed tokens don't get another STS round-trip.
    if let Some(err) = cache.get_negative(password) {
        return Err(err);
    }

    // Precheck failures are free (no I/O) and not negatively cached.
    let prechecked = precheck(password, server_id)?;

    // Only now does the attempt cost an outbound request: take an
    // in-flight slot for the round-trip (released on drop).
    let _permit = inflight_permit()?;

    let client = http_client()?;
    let arn = execute(&client, &prechecked).await;

    finish_verification(
        password,
        arn,
        prechecked.expires_at(),
        server_id,
        allowed,
        cache,
    )
}

/// Turn the outcome of the STS round-trip into an authorization decision,
/// updating the caches: failures (transport, rejection, bad response,
/// disallowed ARN) are negatively cached; only a verified, authorized
/// identity is positively cached.
fn finish_verification(
    password: &str,
    arn: Result<String, StsAuthError>,
    expires_at: SystemTime,
    server_id: &str,
    allowed: &[String],
    cache: &StsTokenCache,
) -> Result<ValidatedStsToken, StsAuthError> {
    let arn = match arn {
        Ok(arn) => arn,
        Err(err) => {
            cache.insert_negative(password, err.clone());
            return Err(err);
        }
    };

    let normalized_arn = normalize_arn(&arn);
    if !matches_allowed(&normalized_arn, &arn, allowed) {
        cache.insert_negative(password, StsAuthError::ArnNotAllowed);
        return Err(StsAuthError::ArnNotAllowed);
    }

    let token = ValidatedStsToken {
        arn,
        normalized_arn,
        expires_at,
        server_id: server_id.to_owned(),
    };
    cache.insert(password, token.clone());

    Ok(token)
}

/// The HTTP client used for STS verification requests.
///
/// Redirects are disabled so the request can only ever reach the
/// prechecked URL. Built once and reused: the client holds a connection
/// pool, so per-verification construction would waste a TLS handshake
/// on every cache miss.
fn http_client() -> Result<reqwest::Client, StsAuthError> {
    static CLIENT: Lazy<Result<reqwest::Client, reqwest::Error>> = Lazy::new(|| {
        reqwest::Client::builder()
            .timeout(STS_REQUEST_TIMEOUT)
            .redirect(redirect::Policy::none())
            .build()
    });

    CLIENT
        .as_ref()
        .map(Clone::clone)
        .map_err(|err| StsAuthError::RequestFailed(err.to_string()))
}

/// Execute the `GetCallerIdentity` request for a prechecked token and
/// return the caller ARN.
async fn execute(
    client: &reqwest::Client,
    token: &PrecheckedToken,
) -> Result<String, StsAuthError> {
    execute_url(client, token.url().clone(), token.server_id()).await
}

/// Execute a `GetCallerIdentity` request against `url` and return the
/// caller ARN.
///
/// Split from [`execute`] so tests can point it at a local mock server;
/// production callers only ever reach it through a [`PrecheckedToken`],
/// whose URL is pinned to a real STS endpoint (host, port and path).
async fn execute_url(
    client: &reqwest::Client,
    url: Url,
    server_id: &str,
) -> Result<String, StsAuthError> {
    // The URL is the client's credential: log its fingerprint, never
    // the URL itself. `reqwest` errors embed the request URL, so it's
    // stripped with `without_url()` before the error is surfaced.
    let token_sha256 = token_fingerprint(url.as_str());

    let mut response = client
        .get(url)
        .header(SERVER_ID_HEADER, server_id)
        .header(ACCEPT, "application/json")
        .send()
        .await
        .map_err(|err| StsAuthError::RequestFailed(err.without_url().to_string()))?;

    let status = response.status();
    if status != StatusCode::OK {
        warn!(
            token_sha256,
            status = status.as_u16(),
            "STS rejected presigned token"
        );
        return Err(StsAuthError::StsRejected(status.as_u16()));
    }

    // Read the body with a hard size cap instead of `text()`.
    let mut body = Vec::new();
    loop {
        let chunk = response
            .chunk()
            .await
            .map_err(|err| StsAuthError::RequestFailed(err.without_url().to_string()))?;
        let Some(chunk) = chunk else { break };
        if body.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(StsAuthError::InvalidResponse(
                "response body too large".into(),
            ));
        }
        body.extend_from_slice(&chunk);
    }
    let body = String::from_utf8(body)
        .map_err(|_| StsAuthError::InvalidResponse("response body is not UTF-8".into()))?;

    parse_caller_identity(&body)
}

/// `GetCallerIdentity` response body as returned by STS for
/// `Accept: application/json`.
#[derive(Deserialize)]
struct CallerIdentityBody {
    #[serde(rename = "GetCallerIdentityResponse")]
    response: CallerIdentityResponse,
}

#[derive(Deserialize)]
struct CallerIdentityResponse {
    #[serde(rename = "GetCallerIdentityResult")]
    result: CallerIdentityResult,
}

#[derive(Deserialize)]
struct CallerIdentityResult {
    #[serde(rename = "Arn")]
    arn: String,
}

/// Extract the caller ARN from an STS JSON `GetCallerIdentityResponse`.
fn parse_caller_identity(body: &str) -> Result<String, StsAuthError> {
    serde_json::from_str::<CallerIdentityBody>(body)
        .map(|body| body.response.result.arn)
        .map_err(|err| StsAuthError::InvalidResponse(err.to_string()))
}

/// Short SHA-256 prefix of the token, safe to log. The token itself is a
/// live credential and must never be logged.
fn token_fingerprint(token: &str) -> String {
    hex::encode(&cache_key(token)[..6])
}

struct CachedValidation {
    token: ValidatedStsToken,
    cache_expires_at: SystemTime,
}

struct CachedRejection {
    error: StsAuthError,
    cache_expires_at: SystemTime,
}

/// Cache of verified STS tokens, keyed by SHA-256 of the full password
/// string, so repeated connections with the same presigned URL skip the
/// round-trip to STS.
///
/// Entries live until the presigned URL expires, capped at [`CACHE_MAX_TTL`]
/// after insertion. Failed verifications are remembered separately for
/// [`NEGATIVE_CACHE_TTL`]. Expired entries in both maps are pruned on
/// access (per key on reads, full sweep on inserts), so neither map grows
/// beyond the distinct tokens seen in one TTL window.
pub struct StsTokenCache {
    inner: Mutex<HashMap<[u8; 32], CachedValidation>>,
    negative: Mutex<HashMap<[u8; 32], CachedRejection>>,
}

impl StsTokenCache {
    fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            negative: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the single global instance.
    pub fn global() -> &'static StsTokenCache {
        static INSTANCE: Lazy<StsTokenCache> = Lazy::new(StsTokenCache::new);
        &INSTANCE
    }

    /// Return the cached validation for this password, if it hasn't expired
    /// and was verified against the same audience (`sts_server_id`).
    pub fn get(&self, password: &str, server_id: &str) -> Option<ValidatedStsToken> {
        self.get_at(password, server_id, SystemTime::now())
    }

    fn get_at(
        &self,
        password: &str,
        server_id: &str,
        now: SystemTime,
    ) -> Option<ValidatedStsToken> {
        let key = cache_key(password);
        let mut inner = self.inner.lock();

        match inner.get(&key) {
            Some(cached)
                if cached.cache_expires_at > now && cached.token.server_id == server_id =>
            {
                Some(cached.token.clone())
            }
            Some(_) => {
                // Prune on access: the entry expired, or `sts_server_id` was
                // rotated since it was verified. A rotated-away entry can
                // never become valid again (the audience header is part of
                // the signature), so drop it and re-verify against STS.
                inner.remove(&key);
                None
            }
            None => None,
        }
    }

    /// Cache a verified token for this password.
    pub fn insert(&self, password: &str, token: ValidatedStsToken) {
        self.insert_at(password, token, SystemTime::now())
    }

    fn insert_at(&self, password: &str, token: ValidatedStsToken, now: SystemTime) {
        let cache_expires_at = token.expires_at.min(now + CACHE_MAX_TTL);
        let mut inner = self.inner.lock();

        // Prune expired entries so abandoned passwords don't accumulate.
        inner.retain(|_, cached| cached.cache_expires_at > now);
        inner.insert(
            cache_key(password),
            CachedValidation {
                token,
                cache_expires_at,
            },
        );
    }

    /// Return the remembered rejection for this token, if it hasn't expired.
    pub fn get_negative(&self, password: &str) -> Option<StsAuthError> {
        self.get_negative_at(password, SystemTime::now())
    }

    fn get_negative_at(&self, password: &str, now: SystemTime) -> Option<StsAuthError> {
        let key = cache_key(password);
        let mut negative = self.negative.lock();

        match negative.get(&key) {
            Some(cached) if cached.cache_expires_at > now => Some(cached.error.clone()),
            Some(_) => {
                // Prune on access.
                negative.remove(&key);
                None
            }
            None => None,
        }
    }

    /// Remember a failed verification for [`NEGATIVE_CACHE_TTL`].
    pub fn insert_negative(&self, password: &str, error: StsAuthError) {
        self.insert_negative_at(password, error, SystemTime::now())
    }

    fn insert_negative_at(&self, password: &str, error: StsAuthError, now: SystemTime) {
        let mut negative = self.negative.lock();

        // Prune expired entries so abandoned tokens don't accumulate.
        negative.retain(|_, cached| cached.cache_expires_at > now);
        negative.insert(
            cache_key(password),
            CachedRejection {
                error,
                cache_expires_at: now + NEGATIVE_CACHE_TTL,
            },
        );
    }
}

fn cache_key(password: &str) -> [u8; 32] {
    let digest = digest::digest(&digest::SHA256, password.as_bytes());
    let mut key = [0u8; 32];
    key.copy_from_slice(digest.as_ref());
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    /// Fixed "now" for deterministic tests: 2026-08-23 12:00:00 UTC.
    fn now() -> SystemTime {
        parse_date("20260823T120000Z")
    }

    fn parse_date(date: &str) -> SystemTime {
        NaiveDateTime::parse_from_str(date, "%Y%m%dT%H%M%SZ")
            .unwrap()
            .and_utc()
            .into()
    }

    fn format_date(t: SystemTime) -> String {
        DateTime::<Utc>::from(t)
            .format("%Y%m%dT%H%M%SZ")
            .to_string()
    }

    /// A presigned URL that passes every precheck rule at `now()`.
    fn valid_url() -> String {
        url_with(
            "sts.us-east-1.amazonaws.com",
            "20260823T115500Z",
            "900",
            "host%3Bx-pgdog-server-id",
        )
    }

    fn url_with(host: &str, date: &str, expires: &str, signed_headers: &str) -> String {
        format!(
            "https://{host}/?Action=GetCallerIdentity&Version=2011-06-15\
             &X-Amz-Algorithm=AWS4-HMAC-SHA256\
             &X-Amz-Credential=AKIAEXAMPLE%2F20260823%2Fus-east-1%2Fsts%2Faws4_request\
             &X-Amz-Date={date}&X-Amz-Expires={expires}\
             &X-Amz-SignedHeaders={signed_headers}\
             &X-Amz-Signature=deadbeef"
        )
    }

    fn check(password: &str) -> Result<PrecheckedToken, StsAuthError> {
        precheck_at(password, "pgdog-example", now())
    }

    // ── precheck: valid URL ─────────────────────────────────────────────────

    #[test]
    fn precheck_accepts_valid_url() {
        let token = check(&valid_url()).unwrap();
        assert_eq!(token.server_id(), "pgdog-example");
        assert_eq!(
            token.expires_at(),
            parse_date("20260823T115500Z") + Duration::from_secs(900)
        );
        assert_eq!(token.url().host_str(), Some("sts.us-east-1.amazonaws.com"));
    }

    #[test]
    fn precheck_accepts_global_sts_endpoint() {
        let url = url_with(
            "sts.amazonaws.com",
            "20260823T115500Z",
            "900",
            "host%3Bx-pgdog-server-id",
        );
        check(&url).unwrap();
    }

    #[test]
    fn public_precheck_accepts_url_dated_now() {
        // The public entry point uses the wall clock; sign the URL "now".
        let url = url_with(
            "sts.us-east-1.amazonaws.com",
            &format_date(SystemTime::now()),
            "900",
            "host%3Bx-pgdog-server-id",
        );
        precheck(&url, "pgdog-example").unwrap();
    }

    // ── precheck: each rule violated ────────────────────────────────────────

    #[test]
    fn precheck_accepts_explicit_default_port_and_root_path() {
        // ":443" is the https default; the URL parser normalizes it away.
        let url = valid_url().replace(
            "sts.us-east-1.amazonaws.com/",
            "sts.us-east-1.amazonaws.com:443/",
        );
        check(&url).unwrap();

        // No path at all normalizes to "/".
        let url = valid_url().replace(".amazonaws.com/?", ".amazonaws.com?");
        check(&url).unwrap();
    }

    #[test]
    fn precheck_rejects_non_url_password() {
        assert_eq!(check("hunter2").unwrap_err(), StsAuthError::InvalidUrl);
    }

    #[test]
    fn precheck_rejects_non_default_port() {
        for port in [":1", ":80", ":8443"] {
            let url = valid_url().replace(
                "sts.us-east-1.amazonaws.com/",
                &format!("sts.us-east-1.amazonaws.com{port}/"),
            );
            assert_eq!(
                check(&url).unwrap_err(),
                StsAuthError::InvalidPort,
                "port {port:?} must be rejected"
            );
        }
    }

    #[test]
    fn precheck_rejects_non_root_path() {
        for path in ["/deep/path", "/x", "//"] {
            let url = valid_url().replace(
                "sts.us-east-1.amazonaws.com/",
                &format!("sts.us-east-1.amazonaws.com{path}"),
            );
            assert_eq!(
                check(&url).unwrap_err(),
                StsAuthError::InvalidPath,
                "path {path:?} must be rejected"
            );
        }
    }

    #[test]
    fn precheck_rejects_http_scheme() {
        let url = valid_url().replace("https://", "http://");
        assert_eq!(check(&url).unwrap_err(), StsAuthError::InvalidScheme);
    }

    #[test]
    fn precheck_rejects_non_sts_hosts() {
        for host in [
            "sts.us-east-1.amazonaws.com.evil.com",
            "evil.com",
            "sts.us_east_1.amazonaws.com",
            "sts.US-EAST-1.amazonaws.com.",
            "sts..amazonaws.com",
            "amazonaws.com",
            "xsts.us-east-1.amazonaws.com",
        ] {
            let url = url_with(host, "20260823T115500Z", "900", "host%3Bx-pgdog-server-id");
            assert_eq!(
                check(&url).unwrap_err(),
                StsAuthError::InvalidHost,
                "host {host:?} must be rejected"
            );
        }
    }

    #[test]
    fn precheck_rejects_userinfo_smuggled_host() {
        let url = "https://sts.amazonaws.com@evil.com/?Action=GetCallerIdentity\
             &X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260823T115500Z\
             &X-Amz-Expires=900&X-Amz-SignedHeaders=host%3Bx-pgdog-server-id";
        assert_eq!(check(url).unwrap_err(), StsAuthError::InvalidHost);
    }

    #[test]
    fn precheck_rejects_wrong_action() {
        let url = valid_url().replace("Action=GetCallerIdentity", "Action=AssumeRole");
        assert_eq!(check(&url).unwrap_err(), StsAuthError::InvalidAction);

        let url = valid_url().replace("Action=GetCallerIdentity&", "");
        assert_eq!(check(&url).unwrap_err(), StsAuthError::InvalidAction);
    }

    #[test]
    fn precheck_rejects_wrong_algorithm() {
        let url = valid_url().replace("AWS4-HMAC-SHA256", "AWS4-HMAC-SHA1");
        assert_eq!(check(&url).unwrap_err(), StsAuthError::InvalidAlgorithm);

        let url = valid_url().replace("&X-Amz-Algorithm=AWS4-HMAC-SHA256", "");
        assert_eq!(check(&url).unwrap_err(), StsAuthError::InvalidAlgorithm);
    }

    #[test]
    fn precheck_rejects_missing_or_non_integer_expires() {
        for expires in ["", "abc", "900.5", "-1"] {
            let url = url_with(
                "sts.us-east-1.amazonaws.com",
                "20260823T115500Z",
                expires,
                "host%3Bx-pgdog-server-id",
            );
            assert_eq!(
                check(&url).unwrap_err(),
                StsAuthError::InvalidExpires,
                "expires {expires:?} must be rejected"
            );
        }

        let url = valid_url().replace("&X-Amz-Expires=900", "");
        assert_eq!(check(&url).unwrap_err(), StsAuthError::InvalidExpires);
    }

    #[test]
    fn precheck_rejects_expires_over_maximum() {
        let url = url_with(
            "sts.us-east-1.amazonaws.com",
            "20260823T115500Z",
            "901",
            "host%3Bx-pgdog-server-id",
        );
        assert_eq!(check(&url).unwrap_err(), StsAuthError::ExpiresTooLong);
    }

    #[test]
    fn precheck_rejects_missing_or_malformed_date() {
        for date in ["", "2026-08-23T11:55:00Z", "20260823T115500", "notadate"] {
            let url = url_with(
                "sts.us-east-1.amazonaws.com",
                date,
                "900",
                "host%3Bx-pgdog-server-id",
            );
            assert_eq!(
                check(&url).unwrap_err(),
                StsAuthError::InvalidDate,
                "date {date:?} must be rejected"
            );
        }

        let url = valid_url().replace("&X-Amz-Date=20260823T115500Z", "");
        assert_eq!(check(&url).unwrap_err(), StsAuthError::InvalidDate);
    }

    #[test]
    fn precheck_rejects_expired_url() {
        // Signed 15 minutes + 1 second before `now()` with a 900s lifetime.
        let url = url_with(
            "sts.us-east-1.amazonaws.com",
            "20260823T114459Z",
            "900",
            "host%3Bx-pgdog-server-id",
        );
        assert_eq!(check(&url).unwrap_err(), StsAuthError::Expired);
    }

    #[test]
    fn precheck_rejects_url_expiring_exactly_now() {
        let url = url_with(
            "sts.us-east-1.amazonaws.com",
            "20260823T114500Z",
            "900",
            "host%3Bx-pgdog-server-id",
        );
        assert_eq!(check(&url).unwrap_err(), StsAuthError::Expired);
    }

    #[test]
    fn precheck_rejects_missing_signed_headers() {
        let url = valid_url().replace("&X-Amz-SignedHeaders=host%3Bx-pgdog-server-id", "");
        assert_eq!(check(&url).unwrap_err(), StsAuthError::MissingSignedHeaders);
    }

    #[test]
    fn precheck_rejects_unsigned_host_header() {
        let url = url_with(
            "sts.us-east-1.amazonaws.com",
            "20260823T115500Z",
            "900",
            "x-pgdog-server-id",
        );
        assert_eq!(check(&url).unwrap_err(), StsAuthError::HostHeaderNotSigned);
    }

    #[test]
    fn precheck_rejects_unsigned_server_id_header() {
        let url = url_with(
            "sts.us-east-1.amazonaws.com",
            "20260823T115500Z",
            "900",
            "host",
        );
        assert_eq!(
            check(&url).unwrap_err(),
            StsAuthError::ServerIdHeaderNotSigned
        );

        // Substring of another header name doesn't count.
        let url = url_with(
            "sts.us-east-1.amazonaws.com",
            "20260823T115500Z",
            "900",
            "host%3Bx-pgdog-server-id-extra",
        );
        assert_eq!(
            check(&url).unwrap_err(),
            StsAuthError::ServerIdHeaderNotSigned
        );
    }

    // ── normalize_arn ───────────────────────────────────────────────────────

    #[test]
    fn normalize_arn_maps_assumed_role_to_iam_role() {
        assert_eq!(
            normalize_arn(
                "arn:aws:sts::123456789012:assumed-role/app-service-role/botocore-session"
            ),
            "arn:aws:iam::123456789012:role/app-service-role"
        );
    }

    #[test]
    fn normalize_arn_leaves_non_matching_arns_unchanged() {
        for arn in [
            "arn:aws:iam::123456789012:role/some-role",
            "arn:aws:iam::123456789012:user/bob",
            "arn:aws:sts::123456789012:federated-user/bob",
            "arn:aws:sts::123456789012:assumed-role/no-session",
            "not-an-arn",
            "",
        ] {
            assert_eq!(normalize_arn(arn), arn, "{arn:?} must be unchanged");
        }
    }

    // ── matches_allowed ─────────────────────────────────────────────────────

    #[test]
    fn matches_allowed_matches_normalized_arn() {
        let allowed = vec!["arn:aws:iam::123456789012:role/some-role".to_string()];
        assert!(matches_allowed(
            "arn:aws:iam::123456789012:role/some-role",
            "arn:aws:sts::123456789012:assumed-role/some-role/session",
            &allowed
        ));
    }

    #[test]
    fn matches_allowed_matches_exact_raw_arn() {
        let allowed = vec!["arn:aws:sts::123456789012:assumed-role/some-role/session".to_string()];
        assert!(matches_allowed(
            "arn:aws:iam::123456789012:role/some-role",
            "arn:aws:sts::123456789012:assumed-role/some-role/session",
            &allowed
        ));
    }

    #[test]
    fn matches_allowed_rejects_non_matching_and_empty_list() {
        let allowed = vec!["arn:aws:iam::123456789012:role/other-role".to_string()];
        assert!(!matches_allowed(
            "arn:aws:iam::123456789012:role/some-role",
            "arn:aws:sts::123456789012:assumed-role/some-role/session",
            &allowed
        ));
        assert!(!matches_allowed(
            "arn:aws:iam::123456789012:role/some-role",
            "arn:aws:sts::123456789012:assumed-role/some-role/session",
            &[]
        ));
    }

    // ── cache ───────────────────────────────────────────────────────────────

    fn token(expires_at: SystemTime) -> ValidatedStsToken {
        ValidatedStsToken {
            arn: "arn:aws:sts::123456789012:assumed-role/some-role/session".into(),
            normalized_arn: "arn:aws:iam::123456789012:role/some-role".into(),
            expires_at,
            server_id: "pgdog-example".into(),
        }
    }

    fn cache() -> StsTokenCache {
        StsTokenCache::new()
    }

    #[test]
    fn cache_returns_inserted_token_before_expiry() {
        let cache = cache();
        let t0 = now();
        cache.insert_at("password-1", token(t0 + Duration::from_secs(900)), t0);

        let cached = cache
            .get_at("password-1", "pgdog-example", t0 + Duration::from_secs(299))
            .unwrap();
        assert_eq!(
            cached.normalized_arn,
            "arn:aws:iam::123456789012:role/some-role"
        );
    }

    #[test]
    fn cache_misses_for_unknown_password() {
        let cache = cache();
        assert!(
            cache
                .get_at("never-inserted", "pgdog-example", now())
                .is_none()
        );
    }

    #[test]
    fn cache_expires_at_insertion_cap_when_token_lives_longer() {
        let cache = cache();
        let t0 = now();
        // Token valid for 900s, but the cache caps at 300s after insertion.
        cache.insert_at("password-2", token(t0 + Duration::from_secs(900)), t0);

        assert!(
            cache
                .get_at("password-2", "pgdog-example", t0 + Duration::from_secs(300))
                .is_none()
        );
    }

    #[test]
    fn cache_expires_at_token_expiry_when_sooner_than_cap() {
        let cache = cache();
        let t0 = now();
        cache.insert_at("password-3", token(t0 + Duration::from_secs(60)), t0);

        assert!(
            cache
                .get_at("password-3", "pgdog-example", t0 + Duration::from_secs(59))
                .is_some()
        );
        assert!(
            cache
                .get_at("password-3", "pgdog-example", t0 + Duration::from_secs(60))
                .is_none()
        );
    }

    #[test]
    fn cache_prunes_expired_entries_on_insert() {
        let cache = cache();
        let t0 = now();
        cache.insert_at("stale", token(t0 + Duration::from_secs(60)), t0);

        // Inserting well past "stale"'s expiry prunes it from the map.
        let t1 = t0 + Duration::from_secs(120);
        cache.insert_at("fresh", token(t1 + Duration::from_secs(60)), t1);

        assert_eq!(cache.inner.lock().len(), 1);
        assert!(cache.get_at("fresh", "pgdog-example", t1).is_some());
    }

    #[test]
    fn cache_keys_by_full_password_string() {
        let cache = cache();
        let t0 = now();
        cache.insert_at("password-a", token(t0 + Duration::from_secs(60)), t0);
        assert!(cache.get_at("password-b", "pgdog-example", t0).is_none());
    }

    #[test]
    fn cache_misses_and_prunes_after_audience_rotation() {
        let cache = cache();
        let t0 = now();
        // Verified while `sts_server_id` was "pgdog-example"...
        cache.insert_at("password-r", token(t0 + Duration::from_secs(900)), t0);
        assert!(cache.get_at("password-r", "pgdog-example", t0).is_some());

        // ...then the audience is rotated: the entry is dead immediately,
        // not after min(token life, CACHE_MAX_TTL).
        assert!(cache.get_at("password-r", "rotated-audience", t0).is_none());
        // And pruned, so it can't come back if the audience rotates again.
        assert!(cache.inner.lock().is_empty());
    }

    #[test]
    fn cache_key_is_sha256_of_password() {
        // SHA-256 of "pgdog", independently computed.
        assert_eq!(
            hex::encode(cache_key("pgdog")),
            "fabb0925b8bb39293ac1260af4e235eb2b3be80dc6e935faef6b834e6615d596"
        );
    }

    #[test]
    fn global_cache_returns_same_instance() {
        let a = StsTokenCache::global() as *const _;
        let b = StsTokenCache::global() as *const _;
        assert_eq!(a, b);
    }

    // ── parse_caller_identity ───────────────────────────────────────────────

    /// The JSON body STS returns for `Accept: application/json`.
    fn caller_identity_json(arn: &str) -> String {
        format!(
            r#"{{"GetCallerIdentityResponse":{{"GetCallerIdentityResult":{{"Account":"123456789012","Arn":"{arn}","UserId":"AROAEXAMPLE:session"}},"ResponseMetadata":{{"RequestId":"01234567-89ab-cdef-0123-456789abcdef"}}}}}}"#
        )
    }

    #[test]
    fn parse_caller_identity_extracts_arn() {
        let arn = "arn:aws:sts::123456789012:assumed-role/some-role/session";
        assert_eq!(
            parse_caller_identity(&caller_identity_json(arn)).unwrap(),
            arn
        );
    }

    #[test]
    fn parse_caller_identity_rejects_missing_arn() {
        let body = r#"{"GetCallerIdentityResponse":{"GetCallerIdentityResult":{"Account":"123456789012"}}}"#;
        assert!(matches!(
            parse_caller_identity(body).unwrap_err(),
            StsAuthError::InvalidResponse(_)
        ));
    }

    #[test]
    fn parse_caller_identity_rejects_non_json_body() {
        // STS answers in XML unless `Accept: application/json` is honored.
        let body = "<GetCallerIdentityResponse></GetCallerIdentityResponse>";
        assert!(matches!(
            parse_caller_identity(body).unwrap_err(),
            StsAuthError::InvalidResponse(_)
        ));
    }

    // ── token_fingerprint ───────────────────────────────────────────────────

    #[test]
    fn token_fingerprint_is_sha256_prefix_not_the_token() {
        // SHA-256 of "pgdog" starts with "fabb0925b8bb".
        let fingerprint = token_fingerprint("pgdog");
        assert_eq!(fingerprint, "fabb0925b8bb");
        assert!(!fingerprint.contains("pgdog"));
    }

    // ── execute_url (mock STS) ──────────────────────────────────────────────

    use wiremock::matchers::{header, method as http_method, path as http_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn setup_tls() {
        let _ = tokio_rustls::rustls::crypto::aws_lc_rs::default_provider().install_default();
    }

    async fn execute_against(server: &MockServer) -> Result<String, StsAuthError> {
        setup_tls();
        let url = Url::parse(&format!("{}/", server.uri())).unwrap();
        let client = http_client().unwrap();
        execute_url(&client, url, "pgdog-example").await
    }

    #[tokio::test]
    async fn execute_url_sends_headers_and_returns_arn() {
        let server = MockServer::start().await;
        let arn = "arn:aws:sts::123456789012:assumed-role/some-role/session";

        // The matchers enforce the request contract: GET to the presigned
        // URL with the audience header and JSON accept header.
        Mock::given(http_method("GET"))
            .and(http_path("/"))
            .and(header(SERVER_ID_HEADER, "pgdog-example"))
            .and(header("accept", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(caller_identity_json(arn)))
            .expect(1)
            .mount(&server)
            .await;

        assert_eq!(execute_against(&server).await.unwrap(), arn);
    }

    #[tokio::test]
    async fn execute_url_maps_non_200_to_sts_rejected() {
        let server = MockServer::start().await;

        Mock::given(http_method("GET"))
            .respond_with(ResponseTemplate::new(403).set_body_string(
                "<ErrorResponse><Error><Code>SignatureDoesNotMatch</Code></Error></ErrorResponse>",
            ))
            .mount(&server)
            .await;

        assert_eq!(
            execute_against(&server).await.unwrap_err(),
            StsAuthError::StsRejected(403)
        );
    }

    #[tokio::test]
    async fn execute_url_does_not_follow_redirects() {
        let server = MockServer::start().await;

        Mock::given(http_method("GET"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "https://evil.com/"))
            .mount(&server)
            .await;

        // Redirects are disabled: a 302 is a rejection, not a request
        // to a new host.
        assert_eq!(
            execute_against(&server).await.unwrap_err(),
            StsAuthError::StsRejected(302)
        );
    }

    #[tokio::test]
    async fn execute_url_maps_invalid_body_to_invalid_response() {
        let server = MockServer::start().await;

        Mock::given(http_method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        assert!(matches!(
            execute_against(&server).await.unwrap_err(),
            StsAuthError::InvalidResponse(_)
        ));
    }

    #[tokio::test]
    async fn execute_url_maps_connect_error_to_request_failed() {
        setup_tls();
        // Port 1 on loopback is closed: the request can't be completed.
        let url = Url::parse("http://127.0.0.1:1/").unwrap();
        let client = http_client().unwrap();

        let err = execute_url(&client, url, "pgdog-example")
            .await
            .unwrap_err();
        match err {
            StsAuthError::RequestFailed(message) => {
                // `reqwest` errors carry the request URL (the token);
                // it must be stripped before the error is surfaced.
                assert!(!message.contains("127.0.0.1"), "leaked URL: {message}");
            }
            other => panic!("expected RequestFailed, got {other:?}"),
        }
    }

    // ── verify ──────────────────────────────────────────────────────────────

    const ALLOWED_ROLE: &str = "arn:aws:iam::123456789012:role/some-role";

    #[tokio::test]
    async fn verify_cache_hit_skips_precheck_and_network() {
        let cache = cache();
        // "cached-token" isn't even a URL: a cache hit must short-circuit
        // before precheck and before any I/O.
        cache.insert(
            "cached-token",
            token(SystemTime::now() + Duration::from_secs(60)),
        );

        let validated = verify(
            "cached-token",
            "pgdog-example",
            &[ALLOWED_ROLE.into()],
            &cache,
        )
        .await
        .unwrap();
        assert_eq!(validated.normalized_arn, ALLOWED_ROLE);
    }

    #[tokio::test]
    async fn verify_cache_hit_reauthorizes_against_callers_list() {
        let cache = cache();
        // Inserted on behalf of a user that allows this role...
        cache.insert(
            "cross-user-token",
            token(SystemTime::now() + Duration::from_secs(60)),
        );

        // ...but the requesting user's own list doesn't.
        let err = verify(
            "cross-user-token",
            "pgdog-example",
            &["arn:aws:iam::123456789012:role/other-role".into()],
            &cache,
        )
        .await
        .unwrap_err();
        assert_eq!(err, StsAuthError::ArnNotAllowed);

        // Empty list rejects too.
        let err = verify("cross-user-token", "pgdog-example", &[], &cache)
            .await
            .unwrap_err();
        assert_eq!(err, StsAuthError::ArnNotAllowed);
    }

    #[tokio::test]
    async fn verify_cache_hit_requires_matching_audience() {
        let cache = cache();
        // Verified under the audience baked into `token()` ("pgdog-example").
        // "rotated-token" isn't a URL, so if the rotated verify below gets
        // past the cache it must fail in precheck — proof the cached verdict
        // wasn't reused.
        cache.insert(
            "rotated-token",
            token(SystemTime::now() + Duration::from_secs(60)),
        );

        // Same audience: served from cache.
        verify(
            "rotated-token",
            "pgdog-example",
            &[ALLOWED_ROLE.into()],
            &cache,
        )
        .await
        .unwrap();

        // `sts_server_id` rotated: the cached verdict no longer applies and
        // the token goes through full verification again.
        let err = verify(
            "rotated-token",
            "rotated-audience",
            &[ALLOWED_ROLE.into()],
            &cache,
        )
        .await
        .unwrap_err();
        assert_eq!(err, StsAuthError::InvalidUrl);
    }

    #[tokio::test]
    async fn verify_propagates_precheck_errors_without_io() {
        let cache = cache();
        let err = verify("hunter2", "pgdog-example", &[ALLOWED_ROLE.into()], &cache)
            .await
            .unwrap_err();
        assert_eq!(err, StsAuthError::InvalidUrl);
        assert!(cache.inner.lock().is_empty());
    }

    // ── in-flight verification cap ──────────────────────────────────────────

    #[tokio::test]
    async fn verify_rejects_when_verification_slots_exhausted() {
        // Take every slot, as if that many verifications were in flight.
        let permits: Vec<_> = (0..MAX_INFLIGHT_VERIFICATIONS)
            .map(|_| INFLIGHT.try_acquire().unwrap())
            .collect();

        // Precheck-valid token: the rejection must come from the cap,
        // after precheck and before any I/O could happen.
        let url = url_with(
            "sts.us-east-1.amazonaws.com",
            &format_date(SystemTime::now()),
            "900",
            "host%3Bx-pgdog-server-id",
        );
        let err = verify(&url, "pgdog-example", &[ALLOWED_ROLE.into()], &cache())
            .await
            .unwrap_err();
        assert_eq!(err, StsAuthError::TooManyVerifications);

        // Releasing the permits frees every slot again.
        drop(permits);
        assert_eq!(INFLIGHT.available_permits(), MAX_INFLIGHT_VERIFICATIONS);
    }

    #[test]
    fn inflight_permit_is_released_on_drop() {
        let permit = inflight_permit().unwrap();
        assert_eq!(INFLIGHT.available_permits(), MAX_INFLIGHT_VERIFICATIONS - 1);

        // The permit is a RAII guard: dropping it (normal return, `?`, or
        // unwinding) restores the slot.
        drop(permit);
        assert_eq!(INFLIGHT.available_permits(), MAX_INFLIGHT_VERIFICATIONS);
    }

    #[tokio::test]
    async fn verify_cache_hits_bypass_the_inflight_cap() {
        let permits: Vec<_> = (0..MAX_INFLIGHT_VERIFICATIONS)
            .map(|_| INFLIGHT.try_acquire().unwrap())
            .collect();

        // Verified tokens keep working even while all verification
        // slots are busy.
        let cache = cache();
        cache.insert("token", token(SystemTime::now() + Duration::from_secs(60)));
        verify("token", "pgdog-example", &[ALLOWED_ROLE.into()], &cache)
            .await
            .unwrap();

        drop(permits);
    }

    // ── negative cache ──────────────────────────────────────────────────────

    #[test]
    fn negative_cache_returns_error_within_ttl_and_expires_after() {
        let cache = cache();
        let t0 = now();
        cache.insert_negative_at("bad-token", StsAuthError::StsRejected(403), t0);

        assert_eq!(
            cache.get_negative_at("bad-token", t0 + Duration::from_secs(29)),
            Some(StsAuthError::StsRejected(403))
        );
        // Expired at the TTL boundary: the client may retry against STS.
        assert_eq!(
            cache.get_negative_at("bad-token", t0 + Duration::from_secs(30)),
            None
        );
    }

    #[test]
    fn negative_cache_prunes_expired_entries_on_insert() {
        let cache = cache();
        let t0 = now();
        cache.insert_negative_at("stale-token", StsAuthError::StsRejected(403), t0);

        // Well past "stale-token"'s expiry: it gets pruned on insert.
        let t1 = t0 + Duration::from_secs(60);
        cache.insert_negative_at("fresh-token", StsAuthError::ArnNotAllowed, t1);

        assert_eq!(cache.negative.lock().len(), 1);
    }

    #[tokio::test]
    async fn verify_short_circuits_on_negative_cache_before_precheck() {
        let cache = cache();
        // Not even a URL: a negative hit must return before precheck runs
        // (and therefore before any I/O could happen).
        cache.insert_negative("bad-token", StsAuthError::StsRejected(403));

        let err = verify("bad-token", "pgdog-example", &[ALLOWED_ROLE.into()], &cache)
            .await
            .unwrap_err();
        assert_eq!(err, StsAuthError::StsRejected(403));
    }

    #[tokio::test]
    async fn verify_positive_cache_wins_over_negative_entry() {
        let cache = cache();
        cache.insert("token", token(SystemTime::now() + Duration::from_secs(60)));
        cache.insert_negative("token", StsAuthError::StsRejected(403));

        // A verified token stays usable: positive entries are checked first,
        // so a negative entry can never shadow a verified identity.
        verify("token", "pgdog-example", &[ALLOWED_ROLE.into()], &cache)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn verify_precheck_failure_does_not_consume_negative_cache_slot() {
        let cache = cache();
        // Precheck failures are free (no I/O), so they must not take up
        // negative-cache memory.
        let err = verify("hunter2", "pgdog-example", &[ALLOWED_ROLE.into()], &cache)
            .await
            .unwrap_err();
        assert_eq!(err, StsAuthError::InvalidUrl);
        assert!(cache.negative.lock().is_empty());
    }

    #[test]
    fn finish_verification_failure_inserts_negative_entry() {
        let cache = cache();
        let expires_at = SystemTime::now() + Duration::from_secs(60);

        let err = finish_verification(
            "rejected-token",
            Err(StsAuthError::StsRejected(403)),
            expires_at,
            "pgdog-example",
            &[ALLOWED_ROLE.into()],
            &cache,
        )
        .unwrap_err();

        assert_eq!(err, StsAuthError::StsRejected(403));
        assert_eq!(
            cache.get_negative("rejected-token"),
            Some(StsAuthError::StsRejected(403))
        );
        assert!(cache.get("rejected-token", "pgdog-example").is_none());
    }

    #[test]
    fn finish_verification_disallowed_arn_inserts_negative_entry() {
        let cache = cache();
        let expires_at = SystemTime::now() + Duration::from_secs(60);
        let arn = "arn:aws:sts::123456789012:assumed-role/other-role/session".to_string();

        let err = finish_verification(
            "wrong-role-token",
            Ok(arn),
            expires_at,
            "pgdog-example",
            &[ALLOWED_ROLE.into()],
            &cache,
        )
        .unwrap_err();

        assert_eq!(err, StsAuthError::ArnNotAllowed);
        assert_eq!(
            cache.get_negative("wrong-role-token"),
            Some(StsAuthError::ArnNotAllowed)
        );
        assert!(cache.get("wrong-role-token", "pgdog-example").is_none());
    }

    #[test]
    fn finish_verification_success_inserts_positive_entry_only() {
        let cache = cache();
        let expires_at = SystemTime::now() + Duration::from_secs(60);
        let arn = "arn:aws:sts::123456789012:assumed-role/some-role/session".to_string();

        let validated = finish_verification(
            "good-token",
            Ok(arn),
            expires_at,
            "pgdog-example",
            &[ALLOWED_ROLE.into()],
            &cache,
        )
        .unwrap();

        assert_eq!(validated.normalized_arn, ALLOWED_ROLE);
        assert!(cache.get("good-token", "pgdog-example").is_some());
        assert_eq!(cache.get_negative("good-token"), None);
    }

    #[tokio::test]
    async fn execute_url_rejects_oversized_body() {
        let server = MockServer::start().await;

        Mock::given(http_method("GET"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string("x".repeat(MAX_RESPONSE_BYTES + 1)),
            )
            .mount(&server)
            .await;

        assert!(matches!(
            execute_against(&server).await.unwrap_err(),
            StsAuthError::InvalidResponse(_)
        ));
    }
}
