//! AWS STS client authentication: presigned GetCallerIdentity URL validation.
//!
//! Clients present a presigned STS `GetCallerIdentity` URL as their Postgres
//! password (the aws-iam-authenticator pattern). This module performs the
//! pure, offline part of that handshake: syntactic validation of the
//! presigned URL, IAM role ARN normalization and matching, and a cache of
//! already-verified tokens. Executing the URL against STS happens elsewhere.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use aws_lc_rs::digest;
use chrono::NaiveDateTime;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use thiserror::Error;
use url::Url;

/// Header clients must include in `X-Amz-SignedHeaders`, carrying the
/// audience value from `sts_server_id` in `pgdog.toml`.
pub const SERVER_ID_HEADER: &str = "x-pgdog-server-id";

/// Maximum allowed value of `X-Amz-Expires`, in seconds.
const MAX_EXPIRES: u64 = 900;

/// Upper bound on how long a verified token stays cached after insertion.
const CACHE_MAX_TTL: Duration = Duration::from_secs(300);

/// Reasons a presigned STS URL fails validation before any I/O happens.
#[derive(Debug, Error, PartialEq, Eq)]
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

struct CachedValidation {
    token: ValidatedStsToken,
    cache_expires_at: SystemTime,
}

/// Cache of verified STS tokens, keyed by SHA-256 of the full password
/// string, so repeated connections with the same presigned URL skip the
/// round-trip to STS.
///
/// Entries live until the presigned URL expires, capped at [`CACHE_MAX_TTL`]
/// after insertion. Expired entries are pruned on access.
pub struct StsTokenCache {
    inner: Mutex<HashMap<[u8; 32], CachedValidation>>,
}

impl StsTokenCache {
    fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the single global instance.
    pub fn global() -> &'static StsTokenCache {
        static INSTANCE: Lazy<StsTokenCache> = Lazy::new(StsTokenCache::new);
        &INSTANCE
    }

    /// Return the cached validation for this password, if it hasn't expired.
    pub fn get(&self, password: &str) -> Option<ValidatedStsToken> {
        self.get_at(password, SystemTime::now())
    }

    fn get_at(&self, password: &str, now: SystemTime) -> Option<ValidatedStsToken> {
        let key = cache_key(password);
        let mut inner = self.inner.lock();

        match inner.get(&key) {
            Some(cached) if cached.cache_expires_at > now => Some(cached.token.clone()),
            Some(_) => {
                // Prune on access.
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
            .get_at("password-1", t0 + Duration::from_secs(299))
            .unwrap();
        assert_eq!(
            cached.normalized_arn,
            "arn:aws:iam::123456789012:role/some-role"
        );
    }

    #[test]
    fn cache_misses_for_unknown_password() {
        let cache = cache();
        assert!(cache.get_at("never-inserted", now()).is_none());
    }

    #[test]
    fn cache_expires_at_insertion_cap_when_token_lives_longer() {
        let cache = cache();
        let t0 = now();
        // Token valid for 900s, but the cache caps at 300s after insertion.
        cache.insert_at("password-2", token(t0 + Duration::from_secs(900)), t0);

        assert!(
            cache
                .get_at("password-2", t0 + Duration::from_secs(300))
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
                .get_at("password-3", t0 + Duration::from_secs(59))
                .is_some()
        );
        assert!(
            cache
                .get_at("password-3", t0 + Duration::from_secs(60))
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
        assert!(cache.get_at("fresh", t1).is_some());
    }

    #[test]
    fn cache_keys_by_full_password_string() {
        let cache = cache();
        let t0 = now();
        cache.insert_at("password-a", token(t0 + Duration::from_secs(60)), t0);
        assert!(cache.get_at("password-b", t0).is_none());
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
}
