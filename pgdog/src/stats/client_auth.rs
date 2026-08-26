//! Client authentication failure counters.
//!
//! Failures are counted by reason so a spike in rejections is visible on
//! the metrics endpoint instead of only in logs. Reasons are for operators
//! only: clients always receive the same uniform authentication error
//! regardless of which check failed.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use super::{Measurement, Metric, OpenMetric};

/// Cumulative failure counts, keyed by [`crate::auth::AuthResult::reason`].
static FAILURES: Lazy<Mutex<HashMap<&'static str, u64>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Record one rejected client authentication attempt.
pub fn record_failure(reason: &'static str) {
    *FAILURES.lock().entry(reason).or_insert(0) += 1;
}

/// Current failure count for one reason.
pub fn failure_count(reason: &str) -> u64 {
    FAILURES.lock().get(reason).copied().unwrap_or(0)
}

pub struct ClientAuth;

impl ClientAuth {
    pub fn load() -> Metric {
        let mut failures: Vec<_> = FAILURES
            .lock()
            .iter()
            .map(|(reason, count)| (*reason, *count))
            .collect();
        // Stable output order for scrapes and tests.
        failures.sort_unstable_by_key(|(reason, _)| *reason);

        let measurements = failures
            .into_iter()
            .map(|(reason, count)| Measurement {
                labels: vec![("reason".into(), reason.into())],
                measurement: count.into(),
            })
            .collect();

        Metric::new(ClientAuthMetric { measurements })
    }
}

struct ClientAuthMetric {
    measurements: Vec<Measurement>,
}

impl OpenMetric for ClientAuthMetric {
    fn name(&self) -> String {
        "client_auth_failures".into()
    }

    fn metric_type(&self) -> String {
        "counter".into()
    }

    fn help(&self) -> Option<String> {
        Some("Total number of rejected client authentication attempts.".into())
    }

    fn measurements(&self) -> Vec<Measurement> {
        self.measurements.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failures_render_as_a_counter_labeled_by_reason() {
        // The map is process-global; use reasons no other test records.
        record_failure("test_reason_b");
        record_failure("test_reason_a");
        record_failure("test_reason_a");

        assert_eq!(failure_count("test_reason_a"), 2);
        assert_eq!(failure_count("test_reason_b"), 1);
        assert_eq!(failure_count("never_recorded"), 0);

        let metric = ClientAuth::load();
        assert_eq!(metric.name(), "client_auth_failures");
        assert_eq!(metric.metric_type(), "counter");

        let rendered = metric.to_string();
        assert!(rendered.contains("client_auth_failures{reason=\"test_reason_a\"} 2"));
        assert!(rendered.contains("client_auth_failures{reason=\"test_reason_b\"} 1"));
    }
}
