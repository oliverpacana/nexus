// crates/nexus-obs/src/tui/theme.rs

use ratatui::style::{Color, Modifier, Style};

/// Color theme for the Nexus TUI dashboard.
#[derive(Debug, Clone, Copy)]
pub struct NexusTheme {
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,
    pub danger: Color,
    pub muted: Color,
    pub background: Color,
    pub border: Color,
    pub text: Color,
    pub title: Style,
    pub status_running: Style,
    pub status_failed: Style,
    pub status_pending: Style,
    pub status_completed: Style,
    pub log_error: Style,
    pub log_warn: Style,
    pub log_info: Style,
    pub log_debug: Style,
}

impl Default for NexusTheme {
    fn default() -> Self {
        Self {
            primary: Color::Cyan,
            secondary: Color::Green,
            accent: Color::Yellow,
            danger: Color::Red,
            muted: Color::Gray,
            background: Color::Black,
            border: Color::DarkGray,
            text: Color::White,
            title: Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            status_running: Style::default().fg(Color::Green),
            status_failed: Style::default().fg(Color::Red),
            status_pending: Style::default().fg(Color::Yellow),
            status_completed: Style::default().fg(Color::Gray),
            log_error: Style::default().fg(Color::Red),
            log_warn: Style::default().fg(Color::Yellow),
            log_info: Style::default().fg(Color::White),
            log_debug: Style::default().fg(Color::Gray),
        }
    }
}

impl NexusTheme {
    /// Returns a style for a given log level string.
    pub fn style_for_level(&self, level: &str) -> Style {
        match level.to_lowercase().as_str() {
            "error" => self.log_error,
            "warn" | "warning" => self.log_warn,
            "info" => self.log_info,
            "debug" | "trace" => self.log_debug,
            _ => self.log_info,
        }
    }

    /// Returns a status icon and style for an agent status string.
    pub fn status_icon(&self, status: &str) -> (&'static str, Style) {
        match status.to_lowercase().as_str() {
            "running" => ("●", self.status_running),
            "failed" => ("✕", self.status_failed),
            "pending" => ("◌", self.status_pending),
            "completed" => ("✓", self.status_completed),
            _ => ("?", Style::default()),
        }
    }
}
