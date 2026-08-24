//! PostgreSQL authentication mechanisms.

pub mod auth_result;
pub mod error;
pub mod md5;
pub mod scram;
// TODO: remove the allow once the STS client-auth handshake is wired up.
#[allow(dead_code)]
pub mod sts;
pub mod vault;

pub use auth_result::AuthResult;
pub use error::Error;
