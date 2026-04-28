// crates/nexus-obs/src/tui/app.rs

use std::collections::VecDeque;
use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Local;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs},
    Terminal,
};
use tracing::{debug, error, info};

use nexus_kernel::KernelHandle;
use nexus_proto::agent::AgentMeta;
use nexus_router::cost::CostRecord;

use crate::error::ObsError;
use crate::ledger::PersistentCostLedger;
use crate::tui::theme::NexusTheme;
use crate::tui::widgets::{cost_table, log_panel, mesh_tree};

/// Configuration for the TUI application.
#[derive(Debug, Clone)]
pub struct TuiConfig {
    /// Refresh rate in milliseconds for UI updates.
    pub refresh_rate_ms: u64,
    /// Handle to the kernel for querying agent state.
    pub kernel_handle: Arc<KernelHandle>,
    /// Persistent cost ledger for financial metrics.
    pub cost_ledger: Arc<PersistentCostLedger>,
}

/// The main TUI application state and event loop.
pub struct TuiApp {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    kernel: Arc<KernelHandle>,
    ledger: Arc<PersistentCostLedger>,
    log_messages: VecDeque<String>,
    selected_tab: usize,    scroll_offset: usize,
    quit: bool,
    refresh_rate: Duration,
    theme: NexusTheme,
    last_refresh: Instant,
}

impl TuiApp {
    /// Creates a new TUI application and initializes the terminal.
    pub fn new(config: TuiConfig) -> Result<Self, ObsError> {
        // Enable raw mode and alternate screen
        enable_raw_mode().map_err(|e| ObsError::Tui(format!("failed to enable raw mode: {}", e)))?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
            .map_err(|e| ObsError::Tui(format!("failed to enter alternate screen: {}", e)))?;

        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)
            .map_err(|e| ObsError::Tui(format!("failed to create terminal: {}", e)))?;

        Ok(Self {
            terminal,
            kernel: config.kernel_handle,
            ledger: config.cost_ledger,
            log_messages: VecDeque::with_capacity(500),
            selected_tab: 0,
            scroll_offset: 0,
            quit: false,
            refresh_rate: Duration::from_millis(config.refresh_rate_ms),
            theme: NexusTheme::default(),
            last_refresh: Instant::now(),
        })
    }

    /// Runs the main TUI event loop until quit is signaled.
    pub async fn run(&mut self) -> Result<(), ObsError> {
        info!("TUI dashboard started");

        while !self.quit {
            // Draw the UI
            self.draw()?;

            // Poll for events with timeout
            if event::poll(self.refresh_rate).map_err(|e| ObsError::Tui(e.to_string()))? {
                if let Event::Key(key) = event::read().map_err(|e| ObsError::Tui(e.to_string()))? {
                    // Only handle key presses, not releases or repeats
                    if key.kind == KeyEventKind::Press {
                        self.handle_key_event(key.code);
                    }
                }            }

            // Periodic refresh of data
            if self.last_refresh.elapsed() >= self.refresh_rate {
                self.refresh_data().await;
                self.last_refresh = Instant::now();
            }
        }

        Ok(())
    }

    /// Handles keyboard input events.
    fn handle_key_event(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Char('Q') => self.quit = true,
            KeyCode::Char('\t') => {
                // Tab to cycle through tabs
                self.selected_tab = (self.selected_tab + 1) % 3;
                self.scroll_offset = 0;
            }
            KeyCode::Char('1') => {
                self.selected_tab = 0;
                self.scroll_offset = 0;
            }
            KeyCode::Char('2') => {
                self.selected_tab = 1;
                self.scroll_offset = 0;
            }
            KeyCode::Char('3') => {
                self.selected_tab = 2;
                self.scroll_offset = 0;
            }
            KeyCode::Up => {
                if self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                }
            }
            KeyCode::Down => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                // Force immediate refresh
                self.last_refresh = Instant::now() - self.refresh_rate;
            }
            _ => {}
        }
    }

    /// Refreshes data from kernel and ledger for the UI.    async fn refresh_data(&mut self) {
        // Fetch agent list
        let agents = self.kernel.list_agents();

        // Fetch recent cost records
        let cost_records = self.ledger.recent_records(50).await.unwrap_or_default();

        // Update log messages (in a real app, this would come from a tracing subscriber)
        // For demo, we add a periodic heartbeat log
        if self.log_messages.len() < 500 {
            let now = Local::now().format("%H:%M:%S");
            self.log_messages.push_back(format!(
                "[{}] INFO: Dashboard refreshed - {} agents, {} cost records",
                now,
                agents.len(),
                cost_records.len()
            ));
        }
    }

    /// Draws the entire UI layout.
    fn draw(&mut self) -> Result<(), ObsError> {
        self.terminal
            .draw(|f| {
                let size = f.area();

                // Top bar
                let top_bar = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Length(20),
                        Constraint::Min(0),
                        Constraint::Length(20),
                    ])
                    .split(size);

                // Title
                f.render_widget(
                    Paragraph::new("⚡ NEXUS")
                        .style(self.theme.title)
                        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(self.theme.border))),
                    top_bar[0],
                );

                // Current time
                let now = Local::now().format("%Y-%m-%d %H:%M:%S");
                f.render_widget(
                    Paragraph::new(format!("{}", now))
                        .style(Style::default().fg(self.theme.muted))
                        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(self.theme.border))),                    top_bar[2],
                );

                // Tab bar
                let tab_titles = vec!["Agents", "Cost", "Logs"];
                let tabs = Tabs::new(tab_titles)
                    .select(self.selected_tab)
                    .style(Style::default().fg(self.theme.muted))
                    .highlight_style(Style::default().fg(self.theme.primary).add_modifier(Modifier::BOLD))
                    .divider(" | ");

                let tab_area = Rect {
                    x: 0,
                    y: 1,
                    width: size.width,
                    height: 1,
                };
                f.render_widget(tabs, tab_area);

                // Main content area
                let main_area = Rect {
                    x: 0,
                    y: 2,
                    width: size.width,
                    height: size.height.saturating_sub(4), // top bar + tabs + status bar
                };

                match self.selected_tab {
                    0 => self.render_agents_tab(f, main_area),
                    1 => self.render_cost_tab(f, main_area),
                    2 => self.render_logs_tab(f, main_area),
                    _ => {}
                }

                // Status bar
                let status_bar = Rect {
                    x: 0,
                    y: size.height.saturating_sub(1),
                    width: size.width,
                    height: 1,
                };

                // Gather status info
                let peer_count = 0; // Would come from mesh node
                let active_agents = self.kernel.stats().active_agents;
                let total_cost = self.ledger.memory_ledger().total_spent();

                let status_text = format!(
                    "Peers: {} | Active: {} | Total Cost: ${:.4} | Press 'q' to quit",
                    peer_count, active_agents, total_cost                );

                f.render_widget(
                    Paragraph::new(status_text)
                        .style(Style::default().fg(self.theme.text))
                        .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(self.theme.border))),
                    status_bar,
                );
            })
            .map_err(|e| ObsError::Tui(format!("failed to draw terminal: {}", e)))
    }

    /// Renders the Agents tab.
    fn render_agents_tab(&self, f: &mut ratatui::Frame, area: Rect) {
        let agents = self.kernel.list_agents();
        mesh_tree::render_agent_tree(f, area, &agents, &self.theme);
    }

    /// Renders the Cost tab.
    fn render_cost_tab(&self, f: &mut ratatui::Frame, area: Rect) {
        // Fetch recent records (synchronous for UI thread; in production, use async channel)
        let records = futures::executor::block_on(self.ledger.recent_records(50))
            .unwrap_or_default();
        cost_table::render_cost_table(f, area, &records, &self.theme);
    }

    /// Renders the Logs tab.
    fn render_logs_tab(&self, f: &mut ratatui::Frame, area: Rect) {
        log_panel::render_log_panel(f, area, &self.log_messages, self.scroll_offset, &self.theme);
    }
}

impl Drop for TuiApp {
    fn drop(&mut self) {
        // Restore terminal state
        disable_raw_mode().ok();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        self.terminal.show_cursor().ok();
        debug!("TUI terminal restored");
    }
}
