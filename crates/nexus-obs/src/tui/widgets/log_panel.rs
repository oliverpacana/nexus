// crates/nexus-obs/src/tui/widgets/log_panel.rs

use std::collections::VecDeque;

use ratatui::{
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};

use crate::tui::theme::NexusTheme;

/// Renders a scrollable log panel with colored levels.
pub fn render_log_panel(
    f: &mut Frame,
    area: Rect,
    messages: &VecDeque<String>,
    scroll_offset: usize,
    theme: &NexusTheme,
) {
    // Convert messages to styled ListItems
    let items: Vec<ListItem> = messages
        .iter()
        .skip(scroll_offset)
        .map(|msg| {
            // Parse log level from message prefix "[LEVEL]"
            let (level, content) = parse_log_level(msg);
            let style = theme.style_for_level(level);

            ListItem::new(vec![Line::from(vec![
                Span::styled(format!("[{}] ", level), style),
                Span::styled(content, Style::default().fg(theme.text)),
            ])])
        })
        .collect();

    // Create list widget
    let list = List::new(items)
        .block(
            Block::default()
                .title(" System Logs ")
                .title_style(theme.title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme.border)),
        )
        .style(Style::default().fg(theme.text))
        .highlight_style(Style::default().add_modifier(ratatui::style::Modifier::REVERSED));

    f.render_widget(list, area);

    // Render scroll indicator if needed
    if scroll_offset > 0 {
        let indicator = Paragraph::new("▲ Scroll up")
            .style(Style::default().fg(theme.muted))
            .block(Block::default().borders(Borders::NONE));
        f.render_widget(indicator, Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: 1,
        });
    }
}

/// Parses a log level from a message string.
/// Expects format: "[LEVEL] message"
fn parse_log_level(msg: &str) -> (&str, &str) {
    if msg.starts_with('[') {
        if let Some(end) = msg.find(']') {
            let level = &msg[1..end];
            let content = if end + 1 < msg.len() {
                &msg[end + 1..].trim()
            } else {
                ""
            };
            (level, content)
        } else {
            ("INFO", msg)
        }
    } else {
        ("INFO", msg)
    }
}
