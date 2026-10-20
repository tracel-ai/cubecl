use super::Text;
use crate::report::OverviewReport;
use std::fmt;

impl fmt::Display for Text<'_, OverviewReport> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            OverviewReport::Summary(report) => fmt::Display::fmt(&Text(report), f),
            OverviewReport::Timeline(report) => fmt::Display::fmt(&Text(report), f),
            OverviewReport::Autotune(report) => fmt::Display::fmt(&Text(report), f),
            OverviewReport::Candidates(report) => fmt::Display::fmt(&Text(report), f),
            OverviewReport::Kernels(report) => fmt::Display::fmt(&Text(report), f),
            OverviewReport::Memory(report) => fmt::Display::fmt(&Text(report), f),
        }
    }
}
