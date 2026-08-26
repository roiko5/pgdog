use std::fmt::Display;

#[derive(Default, PartialEq, Debug, Clone, Copy)]
pub enum AuthResult {
    /// No problems.
    #[default]
    Ok,
    /// Password provided by user doesn't match config.
    NoPasswordMatch,
    /// Passwords not configured.
    NoPasswordConfig,
    /// User identity (TLS cert) doesn't match configured identity.
    NoIdentity,
    /// User requires a client TLS certificate but didn't provide one.
    NoClientCertificate,
    /// Passthrough auth says user doesn't exist.
    NoPassthroughNoUser,
    /// Passthrough auth doesn't allow password changes.
    NoPassthroughPasswordChange,
    /// No user or database in config.
    NoUserOrDatabase,
    /// Client didn't provide password message.
    NoPasswordMessage,
    /// STS token verification failed (malformed, rejected by STS, or the
    /// caller identity isn't in `allowed_iam_arns`).
    NoStsToken,
}

impl AuthResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Stable label for the client auth failure counter, broken down by
    /// reason. Operator-facing only: clients always receive the same
    /// uniform authentication error regardless of which check failed.
    pub fn reason(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::NoPasswordMatch => "wrong_password",
            Self::NoPasswordConfig => "no_password_config",
            Self::NoIdentity => "identity_mismatch",
            Self::NoClientCertificate => "no_client_certificate",
            Self::NoPassthroughNoUser => "passthrough_no_user",
            Self::NoPassthroughPasswordChange => "passthrough_password_change",
            Self::NoUserOrDatabase => "no_user_or_database",
            Self::NoPasswordMessage => "no_password_message",
            Self::NoStsToken => "sts_token_rejected",
        }
    }
}

impl PartialEq<bool> for AuthResult {
    fn eq(&self, other: &bool) -> bool {
        self.is_ok() == *other
    }
}

impl Display for AuthResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "auth ok"),
            Self::NoPasswordMatch => write!(f, "wrong password"),
            Self::NoPasswordConfig => write!(f, "user has no passwords in config"),
            Self::NoIdentity => write!(f, "user identity does not match certificate"),
            Self::NoClientCertificate => {
                write!(
                    f,
                    "user requires a client certificate but none was provided"
                )
            }
            Self::NoPassthroughNoUser => write!(f, "no user in config (passthrough auth)"),
            Self::NoPassthroughPasswordChange => {
                write!(f, "passthrough auth does not allow password change")
            }
            Self::NoUserOrDatabase => write!(f, "no user or database in config"),
            Self::NoPasswordMessage => write!(f, "client did not send password message"),
            Self::NoStsToken => write!(f, "STS token verification failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AuthResult;

    #[test]
    fn sts_rejection_is_an_error_with_its_own_metric_reason() {
        let result = AuthResult::NoStsToken;

        assert!(!result.is_ok());
        // A spike in STS rejections must be distinguishable from a spike
        // in wrong passwords, even though both look identical to clients.
        assert_eq!(result.reason(), "sts_token_rejected");
        assert_ne!(result.reason(), AuthResult::NoPasswordMatch.reason());
    }

    #[test]
    fn no_client_certificate_is_an_error_and_explains_itself() {
        let result = AuthResult::NoClientCertificate;

        assert!(!result.is_ok());
        assert_eq!(
            result.to_string(),
            "user requires a client certificate but none was provided"
        );
    }
}
