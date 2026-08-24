//! Client authentication tests.

use std::time::{Duration, SystemTime};

use pgdog_config::{AuthType, PassthroughAuth};

use crate::{
    auth::sts::{StsTokenCache, ValidatedStsToken},
    backend::databases::reload_from_existing,
    config::{config, set},
    expect_message,
    net::{Authentication, ErrorResponse, Parameters, Password},
};

use super::SpawnedClient;

/// Connect to the admin database and answer the plaintext password
/// request with the given password.
async fn login_admin(password: &str) -> SpawnedClient {
    let cfg = config();
    let mut params = Parameters::default();
    params.insert("user", cfg.config.admin.user.as_str());
    params.insert("database", cfg.config.admin.name.as_str());

    let mut client = SpawnedClient::new_with_login(params).await;

    // Both the admin and the passthrough branches request the password
    // in plaintext; what matters is what happens with the answer.
    let request = expect_message!(client.read().await, Authentication);
    assert!(matches!(request, Authentication::ClearTextPassword));

    client.send(Password::new_password(password)).await;
    client
}

/// Admin connections must be authenticated against the admin password even
/// when passthrough auth is enabled. Regression test for the passthrough
/// branch running first and accepting any password for the admin database.
#[tokio::test]
async fn test_admin_password_checked_with_passthrough_auth() {
    crate::logger();
    crate::config::load_test();

    let mut cfg = (*config()).clone();
    cfg.config.general.auth_type = AuthType::Plain;
    cfg.config.general.passthrough_auth = PassthroughAuth::EnabledPlain;
    cfg.config.admin.password = "admin-password".into();
    set(cfg).unwrap();

    // The wrong password is rejected instead of being passed through.
    let mut client = login_admin("not-the-admin-password").await;
    let error = ErrorResponse::try_from(client.read().await).unwrap();
    assert_eq!(error.code, "28000");
    client.join().await;

    // The correct password is accepted.
    let mut client = login_admin("admin-password").await;
    let response = expect_message!(client.read().await, Authentication);
    assert!(matches!(response, Authentication::Ok));
    client.read_until('Z').await;
    client.join().await;
}

const STS_ALLOWED_ROLE: &str = "arn:aws:iam::123456789012:role/app-service-role";

/// Configure the "pgdog" user for STS client auth, connect, and answer
/// the plaintext password request with the given token.
async fn login_sts(token: &str) -> SpawnedClient {
    crate::logger();
    crate::config::load_test();

    let mut cfg = (*config()).clone();
    cfg.config.general.sts_server_id = Some("pgdog-example".into());
    cfg.users.users[0].allowed_iam_arns = vec![STS_ALLOWED_ROLE.into()];
    // Fail fast after authentication when no Postgres is running: the
    // tests only assert the outcome of the auth exchange.
    cfg.config.general.connect_timeout = 500;
    cfg.config.general.checkout_timeout = 500;
    set(cfg).unwrap();
    reload_from_existing().unwrap();

    let mut params = Parameters::default();
    params.insert("user", "pgdog");
    params.insert("database", "pgdog");

    let mut client = SpawnedClient::new_with_login(params).await;

    // STS client auth requests the token as a plaintext password.
    let request = expect_message!(client.read().await, Authentication);
    assert!(matches!(request, Authentication::ClearTextPassword));

    client.send(Password::new_password(token)).await;
    client
}

/// A user with `allowed_iam_arns` is asked for a plaintext token, and an
/// invalid one is rejected with the same auth error as a wrong password.
#[tokio::test]
async fn test_sts_auth_rejects_invalid_token() {
    let mut client = login_sts("not-a-presigned-sts-url").await;
    let error = ErrorResponse::try_from(client.read().await).unwrap();
    assert_eq!(error.code, "28000");
    client.join().await;
}

/// A token already verified against STS (seeded into the process-wide
/// cache) authenticates the client. This exercises the full handshake
/// wiring without a live STS endpoint.
#[tokio::test]
async fn test_sts_auth_accepts_verified_token() {
    let token = "presigned-sts-token-already-verified";
    StsTokenCache::global().insert(
        token,
        ValidatedStsToken {
            arn: format!(
                "arn:aws:sts::123456789012:assumed-role/{}/session",
                "app-service-role"
            ),
            normalized_arn: STS_ALLOWED_ROLE.into(),
            expires_at: SystemTime::now() + Duration::from_secs(300),
        },
    );

    let mut client = login_sts(token).await;
    // Authentication succeeds. The rest of the login conversation needs a
    // live Postgres pool, so the test stops at the auth outcome.
    let response = expect_message!(client.read().await, Authentication);
    assert!(matches!(response, Authentication::Ok));
}

/// A verified token whose caller identity is not in this user's
/// `allowed_iam_arns` is rejected.
#[tokio::test]
async fn test_sts_auth_rejects_arn_not_in_allowed_list() {
    let token = "presigned-sts-token-wrong-role";
    StsTokenCache::global().insert(
        token,
        ValidatedStsToken {
            arn: "arn:aws:sts::123456789012:assumed-role/other-role/session".into(),
            normalized_arn: "arn:aws:iam::123456789012:role/other-role".into(),
            expires_at: SystemTime::now() + Duration::from_secs(300),
        },
    );

    let mut client = login_sts(token).await;
    let error = ErrorResponse::try_from(client.read().await).unwrap();
    assert_eq!(error.code, "28000");
    client.join().await;
}
