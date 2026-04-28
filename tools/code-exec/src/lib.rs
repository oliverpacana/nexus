// tools/code-exec/src/lib.rs
// WASM tool for sandboxed code execution (Python-like, JS-like, Shell-like, Lua-like)
// Compile with: cargo build --target wasm32-unknown-unknown --release

#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use alloc::boxed::Box;
use core::fmt;
use core::ffi::c_char;
use core::ptr;
use core::str;

// =============================================================================
// Host Function Declarations
// =============================================================================

extern "C" {
    fn nexus_now_ms() -> i64;
    fn nexus_log(level: i32, msg_ptr: *const c_char, msg_len: i32);
}

// =============================================================================
// Static Buffers
// =============================================================================

static mut INPUT_BUFFER: [u8; 65536] = [0u8; 65536];
static mut OUTPUT_BUFFER: [u8; 131072] = [0u8; 131072]; // 128KB for output
static mut INPUT_LEN: u32 = 0;
static mut OUTPUT_LEN: u32 = 0;

// =============================================================================
// ABI Exports
// =============================================================================

#[no_mangle]
pub extern "C" fn nexus_get_input_ptr() -> *mut u8 {
    unsafe { INPUT_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn nexus_get_input_len() -> u32 {
    unsafe { INPUT_LEN }
}

#[no_mangle]
pub unsafe extern "C" fn nexus_set_input_len(len: u32) {    INPUT_LEN = len;
}

#[no_mangle]
pub extern "C" fn nexus_get_output_ptr() -> *mut u8 {
    unsafe { OUTPUT_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn nexus_get_output_len() -> u32 {
    unsafe { OUTPUT_LEN }
}

#[no_mangle]
pub unsafe extern "C" fn nexus_set_output_len(len: u32) {
    OUTPUT_LEN = len;
}

// =============================================================================
// Value Type
// =============================================================================

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Map(Vec<(String, Value)>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Null => write!(f, "None"),
            Value::Bool(b) => write!(f, "{}", if *b { "True" } else { "False" }),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => {
                if fl.fract() == 0.0 {
                    write!(f, "{}.0", *fl as i64)
                } else {
                    write!(f, "{}", fl)
                }
            }
            Value::Str(s) => write!(f, "'{}'", s),
            Value::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "'{}': {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl Value {
    pub fn to_bool(&self) -> bool {
        match self {
            Value::Null | Value::Bool(false) | Value::Int(0) | Value::Float(0.0) | Value::Str(s) if s.is_empty() => false,
            Value::Bool(true) | Value::Int(_) | Value::Float(_) | Value::Str(_) | Value::List(_) | Value::Map(_) => true,
        }
    }

    pub fn to_number(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Str(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Map(_) => "dict",
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < 1e-10,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Map(a), Value::Map(b)) => a == b,
            (Value::Int(a), Value::Float(b)) => (*a as f64 - b).abs() < 1e-10,
            (Value::Float(a), Value::Int(b)) => (a - *b as f64).abs() < 1e-10,
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a.partial_cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Str(a), Value::Str(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

// Arithmetic operators
impl core::ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(a as f64 + b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a + b as f64),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Str(a), Value::Str(b)) => Value::Str(format!("{}{}", a, b)),
            (Value::List(mut a), Value::List(b)) => { a.extend(b); Value::List(a) },
            _ => Value::Null,
        }
    }
}

impl core::ops::Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(a as f64 - b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a - b as f64),            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            _ => Value::Null,
        }
    }
}

impl core::ops::Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(a as f64 * b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a * b as f64),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Str(s), Value::Int(n)) | (Value::Int(n), Value::Str(s)) if n >= 0 => {
                Value::Str(s.repeat(n as usize))
            }
            _ => Value::Null,
        }
    }
}

impl core::ops::Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) if b != 0 => {
                if a % b == 0 {
                    Value::Int(a / b)
                } else {
                    Value::Float(a as f64 / b as f64)
                }
            }
            (Value::Int(a), Value::Float(b)) if b != 0.0 => Value::Float(a as f64 / b),
            (Value::Float(a), Value::Int(b)) if b != 0 => Value::Float(a / b as f64),
            (Value::Float(a), Value::Float(b)) if b != 0.0 => Value::Float(a / b),
            _ => Value::Null,
        }
    }
}

impl core::ops::Rem for Value {
    type Output = Value;
    fn rem(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) if b != 0 => Value::Int(a % b),
            (Value::Float(a), Value::Float(b)) if b != 0.0 => Value::Float(a % b),
            _ => Value::Null,
        }
    }}

// =============================================================================
// Environment
// =============================================================================

pub struct Environment {
    scopes: Vec<alloc::collections::BTreeMap<String, Value>>,
    pub stdout: String,
    pub stderr: String,
    stdin_lines: Vec<String>,
    stdin_pos: usize,
    max_output: usize,
    pub fuel: u64,
    pub exit_code: i32,
}

impl Environment {
    pub fn new(stdin: &str, max_output: usize, fuel: u64) -> Self {
        Self {
            scopes: vec![alloc::collections::BTreeMap::new()],
            stdout: String::new(),
            stderr: String::new(),
            stdin_lines: stdin.lines().map(|s| s.to_string()).collect(),
            stdin_pos: 0,
            max_output,
            fuel,
            exit_code: 0,
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(val);
            }
        }
        None
    }

    pub fn set(&mut self, name: &str, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(alloc::collections::BTreeMap::new());
    }
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    pub fn print(&mut self, s: &str) {
        if self.stdout.len() + s.len() <= self.max_output {
            self.stdout.push_str(s);
            self.stdout.push('\n');
        }
    }

    pub fn eprint(&mut self, s: &str) {
        if self.stderr.len() + s.len() <= self.max_output {
            self.stderr.push_str(s);
            self.stderr.push('\n');
        }
    }

    pub fn consume_fuel(&mut self, amount: u64) -> bool {
        if self.fuel >= amount {
            self.fuel -= amount;
            true
        } else {
            self.fuel = 0;
            false
        }
    }

    pub fn read_line(&mut self) -> Option<String> {
        if self.stdin_pos < self.stdin_lines.len() {
            let line = self.stdin_lines[self.stdin_pos].clone();
            self.stdin_pos += 1;
            Some(line)
        } else {
            None
        }
    }

    pub fn get_all_vars(&self) -> alloc::collections::BTreeMap<String, Value> {
        let mut vars = alloc::collections::BTreeMap::new();
        for scope in &self.scopes {
            for (k, v) in scope {
                vars.insert(k.clone(), v.clone());
            }
        }
        vars
    }
}
// =============================================================================
// Lexer (Tokenizer)
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    None,
    // Operators
    Plus, Minus, Star, Slash, Percent, EqEq, BangEq, Lt, Gt, LtEq, GtEq,
    And, Or, Not, Eq, Colon, Comma, LParen, RParen, LBracket, RBracket,
    LBrace, RBrace, Dot, Arrow,
    // Keywords
    If, Else, Elif, While, For, In, Def, Return, Break, Continue, Pass,
    // Built-ins
    Print, Range, Len, Append, Pop, Keys, Values, Items, Type, Str_, Int_, Float_, Bool_,
    // Special
    Newline, Indent, Dedent, EOF,
}

pub struct Lexer {
    chars: Vec<char>,
    pos: usize,
    indent_stack: Vec<usize>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            chars: source.chars().collect(),
            pos: 0,
            indent_stack: vec![0],
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek();
        self.pos += 1;
        ch
    }
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() && ch != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self, quote: char) -> Option<String> {
        let mut s = String::new();
        while let Some(ch) = self.peek() {
            if ch == quote {
                self.advance();
                return Some(s);
            } else if ch == '\\' {
                self.advance();
                if let Some(escaped) = self.advance() {
                    match escaped {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '\\' => s.push('\\'),
                        '\'' => s.push('\''),
                        '"' => s.push('"'),
                        _ => s.push(escaped),
                    }
                }
            } else {
                s.push(ch);
                self.advance();
            }
        }
        None // Unterminated string
    }

    fn read_number(&mut self) -> Token {
        let mut num_str = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        if num_str.contains('.') {
            Token::Float(num_str.parse().unwrap_or(0.0))
        } else {            Token::Int(num_str.parse().unwrap_or(0))
        }
    }

    fn read_ident(&mut self) -> Token {
        let mut ident = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        match ident.as_str() {
            "True" => Token::Bool(true),
            "False" => Token::Bool(false),
            "None" => Token::None,
            "if" => Token::If,
            "else" => Token::Else,
            "elif" => Token::Elif,
            "while" => Token::While,
            "for" => Token::For,
            "in" => Token::In,
            "def" => Token::Def,
            "return" => Token::Return,
            "break" => Token::Break,
            "continue" => Token::Continue,
            "pass" => Token::Pass,
            "print" => Token::Print,
            "range" => Token::Range,
            "len" => Token::Len,
            "append" => Token::Append,
            "pop" => Token::Pop,
            "keys" => Token::Keys,
            "values" => Token::Values,
            "items" => Token::Items,
            "type" => Token::Type,
            "str" => Token::Str_,
            "int" => Token::Int_,
            "float" => Token::Float_,
            "bool" => Token::Bool_,
            "and" => Token::And,
            "or" => Token::Or,
            "not" => Token::Not,
            _ => Token::Ident(ident),
        }
    }

    pub fn next_token(&mut self) -> Option<Token> {        self.skip_whitespace();
        
        // Handle indentation
        if let Some('\n') = self.peek() {
            self.advance();
            let mut indent = 0;
            while let Some(' ') = self.peek() {
                indent += 1;
                self.advance();
            }
            if let Some('\t') = self.peek() {
                indent += 4; // Treat tab as 4 spaces
                self.advance();
            }
            if indent > *self.indent_stack.last().unwrap_or(&0) {
                self.indent_stack.push(indent);
                return Some(Token::Indent);
            } else if indent < *self.indent_stack.last().unwrap_or(&0) {
                while indent < *self.indent_stack.last().unwrap_or(&0) {
                    self.indent_stack.pop();
                }
                return Some(Token::Dedent);
            }
            return Some(Token::Newline);
        }

        match self.peek()? {
            '#' => {
                // Skip comment
                while let Some(ch) = self.peek() {
                    if ch == '\n' { break; }
                    self.advance();
                }
                self.next_token()
            }
            '+' => { self.advance(); Some(Token::Plus) }
            '-' => { self.advance(); Some(Token::Minus) }
            '*' => { self.advance(); Some(Token::Star) }
            '/' => { self.advance(); Some(Token::Slash) }
            '%' => { self.advance(); Some(Token::Percent) }
            '=' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Some(Token::EqEq)
                } else {
                    Some(Token::Eq)
                }
            }
            '!' => {                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Some(Token::BangEq)
                } else {
                    None
                }
            }
            '<' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Some(Token::LtEq)
                } else {
                    Some(Token::Lt)
                }
            }
            '>' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Some(Token::GtEq)
                } else {
                    Some(Token::Gt)
                }
            }
            '(' => { self.advance(); Some(Token::LParen) }
            ')' => { self.advance(); Some(Token::RParen) }
            '[' => { self.advance(); Some(Token::LBracket) }
            ']' => { self.advance(); Some(Token::RBracket) }
            '{' => { self.advance(); Some(Token::LBrace) }
            '}' => { self.advance(); Some(Token::RBrace) }
            ':' => { self.advance(); Some(Token::Colon) }
            ',' => { self.advance(); Some(Token::Comma) }
            '.' => {
                self.advance();
                if self.peek().map_or(false, |c| c.is_ascii_digit()) {
                    // Float starting with .
                    let mut num = String::from(".");
                    while let Some(ch) = self.peek() {
                        if ch.is_ascii_digit() {
                            num.push(ch);
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    Some(Token::Float(num.parse().unwrap_or(0.0)))
                } else {
                    Some(Token::Dot)                }
            }
            '"' | '\'' => {
                let quote = self.advance().unwrap();
                self.read_string(quote).map(Token::Str)
            }
            ch if ch.is_ascii_digit() => Some(self.read_number()),
            ch if ch.is_alphabetic() || ch == '_' => Some(self.read_ident()),
            _ => { self.advance(); None }
        }
    }
}

// =============================================================================
// Parser (Recursive Descent)
// =============================================================================

#[derive(Clone, Debug)]
pub enum Expr {
    Literal(Value),
    Ident(String),
    BinOp(Box<Expr>, Token, Box<Expr>),
    UnaryOp(Token, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Attr(Box<Expr>, String),
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Assign(String, Expr),
    If(Expr, Vec<Stmt>, Vec<(Expr, Vec<Stmt>)>, Option<Vec<Stmt>>),
    While(Expr, Vec<Stmt>),
    For(String, Expr, Vec<Stmt>),
    FuncDef(String, Vec<String>, Vec<Stmt>),
    Return(Option<Expr>),
    Break,
    Continue,
    Pass,
    ExprStmt(Expr),
    Print(Expr),
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}
impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.peek().cloned();
        self.pos += 1;
        token
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        if self.advance() == Some(expected) {
            Ok(())
        } else {
            Err(format!("Expected {:?}", expected))
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;
        while self.peek() == Some(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::BinOp(Box::new(left), Token::Or, Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_not()?;
        while self.peek() == Some(&Token::And) {
            self.advance();
            let right = self.parse_not()?;
            left = Expr::BinOp(Box::new(left), Token::And, Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, String> {
        if self.peek() == Some(&Token::Not) {
            self.advance();            let expr = self.parse_not()?;
            Ok(Expr::UnaryOp(Token::Not, Box::new(expr)))
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_additive()?;
        while let Some(op) = self.peek() {
            match op {
                Token::EqEq | Token::BangEq | Token::Lt | Token::Gt | Token::LtEq | Token::GtEq | Token::In => {
                    let op = self.advance().unwrap();
                    let right = self.parse_additive()?;
                    left = Expr::BinOp(Box::new(left), op, Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_multiplicative()?;
        while let Some(op) = self.peek() {
            match op {
                Token::Plus | Token::Minus => {
                    let op = self.advance().unwrap();
                    let right = self.parse_multiplicative()?;
                    left = Expr::BinOp(Box::new(left), op, Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_unary()?;
        while let Some(op) = self.peek() {
            match op {
                Token::Star | Token::Slash | Token::Percent => {
                    let op = self.advance().unwrap();
                    let right = self.parse_unary()?;
                    left = Expr::BinOp(Box::new(left), op, Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if let Some(Token::Minus) = self.peek() {
            self.advance();
            let expr = self.parse_unary()?;
            Ok(Expr::UnaryOp(Token::Minus, Box::new(expr)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.advance() {
            Some(Token::Int(i)) => Ok(Expr::Literal(Value::Int(i))),
            Some(Token::Float(f)) => Ok(Expr::Literal(Value::Float(f))),
            Some(Token::Str(s)) => Ok(Expr::Literal(Value::Str(s))),
            Some(Token::Bool(b)) => Ok(Expr::Literal(Value::Bool(b))),
            Some(Token::None) => Ok(Expr::Literal(Value::Null)),
            Some(Token::Ident(name)) => {
                if self.peek() == Some(&Token::LParen) {
                    // Function call
                    self.advance(); // consume '('
                    let mut args = Vec::new();
                    while self.peek() != Some(&Token::RParen) {
                        args.push(self.parse_expr()?);
                        if self.peek() == Some(&Token::Comma) {
                            self.advance();
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Expr::Call(Box::new(Expr::Ident(name)), args))
                } else if self.peek() == Some(&Token::Dot) {
                    // Attribute access
                    self.advance(); // consume '.'
                    if let Some(Token::Ident(attr)) = self.advance() {
                        Ok(Expr::Attr(Box::new(Expr::Ident(name)), attr))
                    } else {
                        Err("Expected attribute name".into())
                    }
                } else if self.peek() == Some(&Token::LBracket) {
                    // Indexing
                    self.advance(); // consume '['
                    let index = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    Ok(Expr::Index(Box::new(Expr::Ident(name)), Box::new(index)))
                } else {
                    Ok(Expr::Ident(name))
                }
            }            Some(Token::LParen) => {
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Some(Token::LBracket) => {
                let mut items = Vec::new();
                while self.peek() != Some(&Token::RBracket) {
                    items.push(self.parse_expr()?);
                    if self.peek() == Some(&Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(Token::RBracket)?;
                Ok(Expr::List(items))
            }
            Some(Token::LBrace) => {
                let mut pairs = Vec::new();
                while self.peek() != Some(&Token::RBrace) {
                    let key = self.parse_expr()?;
                    self.expect(Token::Colon)?;
                    let value = self.parse_expr()?;
                    pairs.push((key, value));
                    if self.peek() == Some(&Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(Token::RBrace)?;
                Ok(Expr::Dict(pairs))
            }
            Some(Token::Print) => {
                self.expect(Token::LParen)?;
                let arg = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(Expr::Call(Box::new(Expr::Ident("print".into())), vec![arg]))
            }
            Some(Token::Range) => {
                self.expect(Token::LParen)?;
                let mut args = Vec::new();
                while self.peek() != Some(&Token::RParen) {
                    args.push(self.parse_expr()?);
                    if self.peek() == Some(&Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(Token::RParen)?;
                Ok(Expr::Call(Box::new(Expr::Ident("range".into())), args))
            }
            Some(Token::Len) => {
                self.expect(Token::LParen)?;                let arg = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(Expr::Call(Box::new(Expr::Ident("len".into())), vec![arg]))
            }
            _ => Err("Unexpected token in expression".into()),
        }
    }

    pub fn parse_stmts(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = Vec::new();
        while self.peek().is_some() && self.peek() != Some(&Token::EOF) {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match self.peek().cloned() {
            Some(Token::If) => {
                self.advance();
                let condition = self.parse_expr()?;
                self.expect(Token::Colon)?;
                let mut then_block = Vec::new();
                while self.peek().map_or(false, |t| matches!(t, Token::Indent | Token::Newline)) {
                    if self.peek() == Some(&Token::Indent) {
                        self.advance();
                    }
                    then_block.push(self.parse_stmt()?);
                    if self.peek() == Some(&Token::Dedent) {
                        self.advance();
                        break;
                    }
                }
                let mut elif_blocks = Vec::new();
                while self.peek() == Some(&Token::Elif) {
                    self.advance();
                    let cond = self.parse_expr()?;
                    self.expect(Token::Colon)?;
                    let mut block = Vec::new();
                    while self.peek().map_or(false, |t| matches!(t, Token::Indent | Token::Newline)) {
                        if self.peek() == Some(&Token::Indent) {
                            self.advance();
                        }
                        block.push(self.parse_stmt()?);
                        if self.peek() == Some(&Token::Dedent) {
                            self.advance();
                            break;
                        }
                    }
                    elif_blocks.push((cond, block));                }
                let else_block = if self.peek() == Some(&Token::Else) {
                    self.advance();
                    self.expect(Token::Colon)?;
                    let mut block = Vec::new();
                    while self.peek().map_or(false, |t| matches!(t, Token::Indent | Token::Newline)) {
                        if self.peek() == Some(&Token::Indent) {
                            self.advance();
                        }
                        block.push(self.parse_stmt()?);
                        if self.peek() == Some(&Token::Dedent) {
                            self.advance();
                            break;
                        }
                    }
                    Some(block)
                } else {
                    None
                };
                Ok(Stmt::If(condition, then_block, elif_blocks, else_block))
            }
            Some(Token::While) => {
                self.advance();
                let condition = self.parse_expr()?;
                self.expect(Token::Colon)?;
                let mut body = Vec::new();
                while self.peek().map_or(false, |t| matches!(t, Token::Indent | Token::Newline)) {
                    if self.peek() == Some(&Token::Indent) {
                        self.advance();
                    }
                    body.push(self.parse_stmt()?);
                    if self.peek() == Some(&Token::Dedent) {
                        self.advance();
                        break;
                    }
                }
                Ok(Stmt::While(condition, body))
            }
            Some(Token::For) => {
                self.advance();
                if let Some(Token::Ident(var)) = self.advance() {
                    self.expect(Token::In)?;
                    let iterable = self.parse_expr()?;
                    self.expect(Token::Colon)?;
                    let mut body = Vec::new();
                    while self.peek().map_or(false, |t| matches!(t, Token::Indent | Token::Newline)) {
                        if self.peek() == Some(&Token::Indent) {
                            self.advance();
                        }
                        body.push(self.parse_stmt()?);                        if self.peek() == Some(&Token::Dedent) {
                            self.advance();
                            break;
                        }
                    }
                    Ok(Stmt::For(var, iterable, body))
                } else {
                    Err("Expected variable name in for loop".into())
                }
            }
            Some(Token::Def) => {
                self.advance();
                if let Some(Token::Ident(name)) = self.advance() {
                    self.expect(Token::LParen)?;
                    let mut params = Vec::new();
                    while self.peek() != Some(&Token::RParen) {
                        if let Some(Token::Ident(param)) = self.advance() {
                            params.push(param);
                        }
                        if self.peek() == Some(&Token::Comma) {
                            self.advance();
                        }
                    }
                    self.expect(Token::RParen)?;
                    self.expect(Token::Colon)?;
                    let mut body = Vec::new();
                    while self.peek().map_or(false, |t| matches!(t, Token::Indent | Token::Newline)) {
                        if self.peek() == Some(&Token::Indent) {
                            self.advance();
                        }
                        body.push(self.parse_stmt()?);
                        if self.peek() == Some(&Token::Dedent) {
                            self.advance();
                            break;
                        }
                    }
                    Ok(Stmt::FuncDef(name, params, body))
                } else {
                    Err("Expected function name".into())
                }
            }
            Some(Token::Return) => {
                self.advance();
                let expr = if self.peek().map_or(true, |t| matches!(t, Token::Newline | Token::EOF | Token::Dedent)) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                Ok(Stmt::Return(expr))
            }            Some(Token::Break) => { self.advance(); Ok(Stmt::Break) }
            Some(Token::Continue) => { self.advance(); Ok(Stmt::Continue) }
            Some(Token::Pass) => { self.advance(); Ok(Stmt::Pass) }
            Some(Token::Print) => {
                self.advance();
                self.expect(Token::LParen)?;
                let arg = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(Stmt::Print(arg))
            }
            Some(Token::Ident(name)) => {
                if self.peek() == Some(&Token::Eq) {
                    self.advance();
                    let value = self.parse_expr()?;
                    Ok(Stmt::Assign(name, value))
                } else {
                    let expr = self.parse_expr()?;
                    Ok(Stmt::ExprStmt(expr))
                }
            }
            _ => {
                let expr = self.parse_expr()?;
                Ok(Stmt::ExprStmt(expr))
            }
        }
    }
}

// =============================================================================
// Evaluator
// =============================================================================

pub struct Interpreter {
    env: Environment,
    functions: alloc::collections::BTreeMap<String, (Vec<String>, Vec<Stmt>)>,
}

impl Interpreter {
    pub fn new(env: Environment) -> Self {
        Self {
            env,
            functions: alloc::collections::BTreeMap::new(),
        }
    }

    pub fn eval_stmts(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for stmt in stmts {
            self.eval_stmt(stmt)?;
            if self.env.exit_code != 0 {
                break;            }
        }
        Ok(())
    }

    fn eval_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        if !self.env.consume_fuel(10) {
            return Err("Out of fuel".into());
        }

        match stmt {
            Stmt::Assign(name, expr) => {
                let value = self.eval_expr(expr)?;
                self.env.set(name, value);
            }
            Stmt::If(condition, then_block, elif_blocks, else_block) => {
                if self.eval_expr(condition)?.to_bool() {
                    self.eval_stmts(then_block)?;
                } else {
                    let mut handled = false;
                    for (cond, block) in elif_blocks {
                        if self.eval_expr(cond)?.to_bool() {
                            self.eval_stmts(block)?;
                            handled = true;
                            break;
                        }
                    }
                    if !handled {
                        if let Some(else_blk) = else_block {
                            self.eval_stmts(else_blk)?;
                        }
                    }
                }
            }
            Stmt::While(condition, body) => {
                while self.eval_expr(condition)?.to_bool() {
                    self.env.push_scope();
                    let result = self.eval_stmts(body);
                    self.env.pop_scope();
                    if let Err(e) = result {
                        if e == "break" {
                            break;
                        } else if e == "continue" {
                            continue;
                        } else {
                            return Err(e);
                        }
                    }
                    if !self.env.consume_fuel(100) {
                        return Err("Loop out of fuel".into());                    }
                }
            }
            Stmt::For(var, iterable, body) => {
                let iter_value = self.eval_expr(iterable)?;
                match iter_value {
                    Value::List(items) => {
                        for item in items {
                            self.env.push_scope();
                            self.env.set(var, item);
                            let result = self.eval_stmts(body);
                            self.env.pop_scope();
                            if let Err(e) = result {
                                if e == "break" {
                                    break;
                                } else if e == "continue" {
                                    continue;
                                } else {
                                    return Err(e);
                                }
                            }
                            if !self.env.consume_fuel(100) {
                                return Err("Loop out of fuel".into());
                            }
                        }
                    }
                    _ => return Err("Can only iterate over lists".into()),
                }
            }
            Stmt::FuncDef(name, params, body) => {
                self.functions.insert(name.clone(), (params.clone(), body.clone()));
            }
            Stmt::Return(expr) => {
                let value = if let Some(e) = expr {
                    self.eval_expr(e)?
                } else {
                    Value::Null
                };
                // Return value is handled by caller via Result
                return Err("return".into());
            }
            Stmt::Break => return Err("break".into()),
            Stmt::Continue => return Err("continue".into()),
            Stmt::Pass => {}
            Stmt::ExprStmt(expr) => {
                self.eval_expr(expr)?;
            }
            Stmt::Print(expr) => {
                let value = self.eval_expr(expr)?;
                self.env.print(&format!("{}", value));            }
        }
        Ok(())
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, String> {
        if !self.env.consume_fuel(5) {
            return Err("Out of fuel".into());
        }

        match expr {
            Expr::Literal(v) => Ok(v.clone()),
            Expr::Ident(name) => {
                if let Some(val) = self.env.get(name) {
                    Ok(val.clone())
                } else if let Some((params, body)) = self.functions.get(name) {
                    // Function call without args
                    self.env.push_scope();
                    for param in params {
                        self.env.set(param, Value::Null);
                    }
                    let result = self.eval_stmts(body);
                    self.env.pop_scope();
                    match result {
                        Ok(_) => Ok(Value::Null),
                        Err(e) if e == "return" => Ok(Value::Null), // Simplified
                        Err(e) => Err(e),
                    }
                } else {
                    Err(format!("Undefined variable: {}", name))
                }
            }
            Expr::BinOp(left, op, right) => {
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                match op {
                    Token::Plus => Ok(l + r),
                    Token::Minus => Ok(l - r),
                    Token::Star => Ok(l * r),
                    Token::Slash => Ok(l / r),
                    Token::Percent => Ok(l % r),
                    Token::EqEq => Ok(Value::Bool(l == r)),
                    Token::BangEq => Ok(Value::Bool(l != r)),
                    Token::Lt => Ok(Value::Bool(l < r)),
                    Token::Gt => Ok(Value::Bool(l > r)),
                    Token::LtEq => Ok(Value::Bool(l <= r)),
                    Token::GtEq => Ok(Value::Bool(l >= r)),
                    Token::And => Ok(Value::Bool(l.to_bool() && r.to_bool())),
                    Token::Or => Ok(Value::Bool(l.to_bool() || r.to_bool())),
                    Token::In => {                        match (l, r) {
                            (Value::Str(s), Value::Str(list)) => Ok(Value::Bool(list.contains(&s))),
                            (item, Value::List(list)) => Ok(Value::Bool(list.contains(&item))),
                            _ => Ok(Value::Bool(false)),
                        }
                    }
                    _ => Err(format!("Unsupported operator: {:?}", op)),
                }
            }
            Expr::UnaryOp(op, expr) => {
                let val = self.eval_expr(expr)?;
                match op {
                    Token::Minus => {
                        match val {
                            Value::Int(i) => Ok(Value::Int(-i)),
                            Value::Float(f) => Ok(Value::Float(-f)),
                            _ => Err("Unary minus requires number".into()),
                        }
                    }
                    Token::Not => Ok(Value::Bool(!val.to_bool())),
                    _ => Err(format!("Unsupported unary operator: {:?}", op)),
                }
            }
            Expr::Call(func, args) => {
                let func_val = self.eval_expr(func)?;
                if let Expr::Ident(name) = func.as_ref() {
                    // Built-in functions
                    match name.as_str() {
                        "print" => {
                            for arg in args {
                                let val = self.eval_expr(arg)?;
                                self.env.print(&format!("{}", val));
                            }
                            Ok(Value::Null)
                        }
                        "len" => {
                            if args.len() != 1 {
                                return Err("len() takes exactly one argument".into());
                            }
                            match self.eval_expr(&args[0])? {
                                Value::Str(s) => Ok(Value::Int(s.chars().count() as i64)),
                                Value::List(l) => Ok(Value::Int(l.len() as i64)),
                                Value::Map(m) => Ok(Value::Int(m.len() as i64)),
                                _ => Err("len() requires string, list, or dict".into()),
                            }
                        }
                        "range" => {
                            let mut start = 0;
                            let mut end = 0;
                            let mut step = 1;                            if args.len() == 1 {
                                end = self.eval_expr(&args[0])?.to_number().unwrap_or(0.0) as i64;
                            } else if args.len() == 2 {
                                start = self.eval_expr(&args[0])?.to_number().unwrap_or(0.0) as i64;
                                end = self.eval_expr(&args[1])?.to_number().unwrap_or(0.0) as i64;
                            } else if args.len() == 3 {
                                start = self.eval_expr(&args[0])?.to_number().unwrap_or(0.0) as i64;
                                end = self.eval_expr(&args[1])?.to_number().unwrap_or(0.0) as i64;
                                step = self.eval_expr(&args[2])?.to_number().unwrap_or(1.0) as i64;
                            } else {
                                return Err("range() takes 1-3 arguments".into());
                            }
                            let mut items = Vec::new();
                            if step > 0 {
                                let mut i = start;
                                while i < end {
                                    items.push(Value::Int(i));
                                    i += step;
                                }
                            } else if step < 0 {
                                let mut i = start;
                                while i > end {
                                    items.push(Value::Int(i));
                                    i += step;
                                }
                            }
                            Ok(Value::List(items))
                        }
                        "str" => {
                            if args.len() != 1 {
                                return Err("str() takes exactly one argument".into());
                            }
                            let val = self.eval_expr(&args[0])?;
                            Ok(Value::Str(format!("{}", val)))
                        }
                        "int" => {
                            if args.len() != 1 {
                                return Err("int() takes exactly one argument".into());
                            }
                            let val = self.eval_expr(&args[0])?;
                            match val {
                                Value::Str(s) => {
                                    if let Ok(i) = s.parse::<i64>() {
                                        Ok(Value::Int(i))
                                    } else {
                                        Err(format!("Cannot convert '{}' to int", s))
                                    }
                                }
                                Value::Float(f) => Ok(Value::Int(f as i64)),
                                Value::Int(i) => Ok(Value::Int(i)),                                _ => Err(format!("Cannot convert {} to int", val.type_name())),
                            }
                        }
                        "float" => {
                            if args.len() != 1 {
                                return Err("float() takes exactly one argument".into());
                            }
                            let val = self.eval_expr(&args[0])?;
                            match val {
                                Value::Str(s) => {
                                    if let Ok(f) = s.parse::<f64>() {
                                        Ok(Value::Float(f))
                                    } else {
                                        Err(format!("Cannot convert '{}' to float", s))
                                    }
                                }
                                Value::Int(i) => Ok(Value::Float(i as f64)),
                                Value::Float(f) => Ok(Value::Float(f)),
                                _ => Err(format!("Cannot convert {} to float", val.type_name())),
                            }
                        }
                        "bool" => {
                            if args.len() != 1 {
                                return Err("bool() takes exactly one argument".into());
                            }
                            let val = self.eval_expr(&args[0])?;
                            Ok(Value::Bool(val.to_bool()))
                        }
                        "type" => {
                            if args.len() != 1 {
                                return Err("type() takes exactly one argument".into());
                            }
                            let val = self.eval_expr(&args[0])?;
                            Ok(Value::Str(val.type_name().to_string()))
                        }
                        "input" => {
                            if let Some(line) = self.env.read_line() {
                                Ok(Value::Str(line))
                            } else {
                                Ok(Value::Str("".into()))
                            }
                        }
                        "abs" => {
                            if args.len() != 1 {
                                return Err("abs() takes exactly one argument".into());
                            }
                            let val = self.eval_expr(&args[0])?;
                            match val {
                                Value::Int(i) => Ok(Value::Int(i.abs())),
                                Value::Float(f) => Ok(Value::Float(f.abs())),                                _ => Err("abs() requires number".into()),
                            }
                        }
                        "min" | "max" => {
                            if args.is_empty() {
                                return Err(format!("{}() requires at least one argument", name));
                            }
                            let mut values = Vec::new();
                            for arg in args {
                                values.push(self.eval_expr(arg)?);
                            }
                            if let Some(first) = values.first() {
                                let mut best = first.clone();
                                for val in &values[1..] {
                                    if name == "min" {
                                        if val < &best {
                                            best = val.clone();
                                        }
                                    } else {
                                        if val > &best {
                                            best = val.clone();
                                        }
                                    }
                                }
                                Ok(best)
                            } else {
                                Err(format!("{}() requires arguments", name))
                            }
                        }
                        "sum" => {
                            if args.len() != 1 {
                                return Err("sum() takes exactly one argument".into());
                            }
                            match self.eval_expr(&args[0])? {
                                Value::List(items) => {
                                    let mut total = 0.0;
                                    for item in items {
                                        if let Some(num) = item.to_number() {
                                            total += num;
                                        } else {
                                            return Err("sum() requires list of numbers".into());
                                        }
                                    }
                                    Ok(Value::Float(total))
                                }
                                _ => Err("sum() requires a list".into()),
                            }
                        }
                        "sorted" => {
                            if args.len() != 1 {                                return Err("sorted() takes exactly one argument".into());
                            }
                            match self.eval_expr(&args[0])? {
                                Value::List(mut items) => {
                                    items.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                                    Ok(Value::List(items))
                                }
                                _ => Err("sorted() requires a list".into()),
                            }
                        }
                        "reversed" => {
                            if args.len() != 1 {
                                return Err("reversed() takes exactly one argument".into());
                            }
                            match self.eval_expr(&args[0])? {
                                Value::List(mut items) => {
                                    items.reverse();
                                    Ok(Value::List(items))
                                }
                                _ => Err("reversed() requires a list".into()),
                            }
                        }
                        "enumerate" => {
                            if args.len() != 1 {
                                return Err("enumerate() takes exactly one argument".into());
                            }
                            match self.eval_expr(&args[0])? {
                                Value::List(items) => {
                                    let mut result = Vec::new();
                                    for (i, item) in items.into_iter().enumerate() {
                                        result.push(Value::List(vec![Value::Int(i as i64), item]));
                                    }
                                    Ok(Value::List(result))
                                }
                                _ => Err("enumerate() requires a list".into()),
                            }
                        }
                        "zip" => {
                            if args.len() < 2 {
                                return Err("zip() requires at least two arguments".into());
                            }
                            let mut lists = Vec::new();
                            for arg in args {
                                match self.eval_expr(arg)? {
                                    Value::List(l) => lists.push(l),
                                    _ => return Err("zip() requires lists".into()),
                                }
                            }
                            let min_len = lists.iter().map(|l| l.len()).min().unwrap_or(0);
                            let mut result = Vec::new();                            for i in 0..min_len {
                                let mut tuple = Vec::new();
                                for list in &lists {
                                    tuple.push(list[i].clone());
                                }
                                result.push(Value::List(tuple));
                            }
                            Ok(Value::List(result))
                        }
                        _ => Err(format!("Unknown built-in function: {}", name)),
                    }
                } else {
                    Err("Only built-in functions are supported".into())
                }
            }
            Expr::Index(obj, index) => {
                let obj_val = self.eval_expr(obj)?;
                let idx_val = self.eval_expr(index)?;
                match (obj_val, idx_val) {
                    (Value::List(list), Value::Int(i)) => {
                        let idx = if i < 0 { list.len() as i64 + i } else { i };
                        if idx >= 0 && (idx as usize) < list.len() {
                            Ok(list[idx as usize].clone())
                        } else {
                            Err("List index out of range".into())
                        }
                    }
                    (Value::Str(s), Value::Int(i)) => {
                        let idx = if i < 0 { s.chars().count() as i64 + i } else { i };
                        if idx >= 0 && (idx as usize) < s.chars().count() {
                            Ok(Value::Str(s.chars().nth(idx as usize).unwrap().to_string()))
                        } else {
                            Err("String index out of range".into())
                        }
                    }
                    (Value::Map(map), Value::Str(key)) => {
                        if let Some(val) = map.iter().find(|(k, _)| k == &key).map(|(_, v)| v.clone()) {
                            Ok(val)
                        } else {
                            Err(format!("Key '{}' not found in dict", key))
                        }
                    }
                    _ => Err("Indexing only supported for lists, strings, and dicts".into()),
                }
            }
            Expr::Attr(obj, attr) => {
                let obj_val = self.eval_expr(obj)?;
                match (obj_val, attr.as_str()) {
                    (Value::List(list), "append") => {
                        // Return a closure-like value that will append when called                        // Simplified: we'll handle .append(x) as a special case in Call
                        Ok(Value::Str("append_method".into()))
                    }
                    (Value::List(list), "pop") => {
                        Ok(Value::Str("pop_method".into()))
                    }
                    (Value::Str(s), method) => {
                        // String methods
                        match method {
                            "upper" => Ok(Value::Str(s.to_uppercase())),
                            "lower" => Ok(Value::Str(s.to_lowercase())),
                            "strip" => Ok(Value::Str(s.trim().to_string())),
                            "split" => {
                                // split() with no args splits on whitespace
                                Ok(Value::List(s.split_whitespace().map(|s| Value::Str(s.to_string())).collect()))
                            }
                            "join" => {
                                // join takes a list of strings
                                // We'll handle this in the Call evaluator
                                Ok(Value::Str("join_method".into()))
                            }
                            "replace" => Ok(Value::Str("replace_method".into())),
                            "startswith" => Ok(Value::Str("startswith_method".into())),
                            "endswith" => Ok(Value::Str("endswith_method".into())),
                            "find" => Ok(Value::Str("find_method".into())),
                            "format" => Ok(Value::Str("format_method".into())),
                            _ => Err(format!("String method '{}' not supported", method)),
                        }
                    }
                    (Value::Map(map), method) => {
                        match method {
                            "keys" => Ok(Value::List(map.iter().map(|(k, _)| Value::Str(k.clone())).collect())),
                            "values" => Ok(Value::List(map.iter().map(|(_, v)| v.clone()).collect())),
                            "items" => {
                                let items = map.iter().map(|(k, v)| {
                                    Value::List(vec![Value::Str(k.clone()), v.clone()])
                                }).collect();
                                Ok(Value::List(items))
                            }
                            "get" => Ok(Value::Str("get_method".into())),
                            "update" => Ok(Value::Str("update_method".into())),
                            _ => Err(format!("Dict method '{}' not supported", method)),
                        }
                    }
                    _ => Err(format!("Attribute '{}' not supported for type {}", attr, obj_val.type_name())),
                }
            }
            Expr::List(items) => {
                let mut values = Vec::new();
                for item in items {                    values.push(self.eval_expr(item)?);
                }
                Ok(Value::List(values))
            }
            Expr::Dict(pairs) => {
                let mut map = Vec::new();
                for (key_expr, value_expr) in pairs {
                    let key = self.eval_expr(key_expr)?;
                    if let Value::Str(key_str) = key {
                        let value = self.eval_expr(value_expr)?;
                        map.push((key_str, value));
                    } else {
                        return Err("Dict keys must be strings".into());
                    }
                }
                Ok(Value::Map(map))
            }
        }
    }
}

// =============================================================================
// Shell-like Evaluator (Minimal)
// =============================================================================

fn eval_shell(code: &str, env: &mut Environment) -> Result<(), String> {
    // Very basic shell: split by ; and execute each command
    for cmd in code.split(';') {
        let cmd = cmd.trim();
        if cmd.is_empty() { continue; }
        
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() { continue; }
        
        match parts[0] {
            "echo" => {
                let output = parts[1..].join(" ");
                env.print(&output);
            }
            "printf" => {
                if parts.len() < 2 {
                    env.eprint("printf: missing format string");
                    env.exit_code = 1;
                    continue;
                }
                let format_str = parts[1];
                let args = &parts[2..];
                // Very basic printf: only %s and %d
                let mut result = String::new();
                let mut arg_idx = 0;                let mut chars = format_str.chars().peekable();
                while let Some(ch) = chars.next() {
                    if ch == '%' {
                        if let Some(spec) = chars.next() {
                            match spec {
                                's' => {
                                    if arg_idx < args.len() {
                                        result.push_str(args[arg_idx]);
                                        arg_idx += 1;
                                    }
                                }
                                'd' => {
                                    if arg_idx < args.len() {
                                        if let Ok(num) = args[arg_idx].parse::<i64>() {
                                            result.push_str(&num.to_string());
                                        }
                                        arg_idx += 1;
                                    }
                                }
                                '%' => result.push('%'),
                                _ => result.push('%').push(spec),
                            }
                        }
                    } else {
                        result.push(ch);
                    }
                }
                env.print(&result);
            }
            "expr" => {
                // Basic arithmetic: expr 2 + 3
                if parts.len() != 4 {
                    env.eprint("expr: expected three arguments");
                    env.exit_code = 1;
                    continue;
                }
                let a = parts[1].parse::<i64>().unwrap_or(0);
                let op = parts[2];
                let b = parts[3].parse::<i64>().unwrap_or(0);
                let result = match op {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => if b != 0 { a / b } else { 0 },
                    "%" => if b != 0 { a % b } else { 0 },
                    _ => {
                        env.eprint(&format!("expr: unknown operator {}", op));
                        env.exit_code = 1;
                        continue;
                    }                };
                env.print(&result.to_string());
            }
            "test" | "[" => {
                // Very basic test: test 1 -eq 1 or [ 1 -eq 1 ]
                // Supports: -eq, -ne, -lt, -gt, -z, -n
                let args = if parts[0] == "[" { &parts[1..parts.len()-1] } else { &parts[1..] };
                if args.len() == 3 {
                    let a = args[0];
                    let op = args[1];
                    let b = args[2];
                    let success = match op {
                        "-eq" => a.parse::<i64>().unwrap_or(0) == b.parse::<i64>().unwrap_or(0),
                        "-ne" => a.parse::<i64>().unwrap_or(0) != b.parse::<i64>().unwrap_or(0),
                        "-lt" => a.parse::<i64>().unwrap_or(0) < b.parse::<i64>().unwrap_or(0),
                        "-gt" => a.parse::<i64>().unwrap_or(0) > b.parse::<i64>().unwrap_or(0),
                        "-z" => a.is_empty(),
                        "-n" => !a.is_empty(),
                        _ => false,
                    };
                    if !success {
                        env.exit_code = 1;
                    }
                } else if args.len() == 1 {
                    // test -z string or test string
                    if args[0] == "-z" {
                        env.exit_code = 1; // Always false without argument
                    } else if args[0] == "-n" {
                        env.exit_code = 0; // Always true without argument
                    } else {
                        env.exit_code = if args[0].is_empty() { 1 } else { 0 };
                    }
                }
            }
            "true" => { env.exit_code = 0; }
            "false" => { env.exit_code = 1; }
            "export" => {
                // export KEY=VALUE
                if parts.len() == 2 {
                    if let Some(eq_pos) = parts[1].find('=') {
                        let key = &parts[1][..eq_pos];
                        let value = &parts[1][eq_pos+1..];
                        env.set(key, Value::Str(value.to_string()));
                    }
                }
            }
            _ => {
                env.eprint(&format!("Unknown command: {}", parts[0]));
                env.exit_code = 127;
            }        }
    }
    Ok(())
}

// =============================================================================
// Main Entry Point
// =============================================================================

#[no_mangle]
pub extern "C" fn _nexus_run() -> i32 {
    let start_time = unsafe { nexus_now_ms() };
    
    // Parse input
    let input_json = unsafe {
        let ptr = nexus_get_input_ptr();
        let len = nexus_get_input_len() as usize;
        if len == 0 || len > INPUT_BUFFER.len() {
            write_error("Invalid input length");
            return 1;
        }
        str::from_utf8(&INPUT_BUFFER[..len]).unwrap_or("")
    };
    
    let code = match parse_json_string(input_json, "code") {
        Some(c) => c,
        None => {
            write_error("Missing 'code' field in input");
            return 1;
        }
    };
    
    let language = match parse_json_string(input_json, "language") {
        Some(l) => l,
        None => {
            write_error("Missing 'language' field in input");
            return 1;
        }
    };
    
    let stdin = parse_json_string(input_json, "stdin").unwrap_or_default();
    let timeout_ms = parse_json_int(input_json, "timeout_ms", 2000) as u64;
    let max_output = parse_json_int(input_json, "max_output_bytes", 16384) as usize;
    
    // Set up environment
    let mut env = Environment::new(&stdin, max_output, timeout_ms * 1000); // Convert ms to fuel units
    
    // Execute based on language
    let result = match language.as_str() {
        "python" | "python-like" => {            let mut lexer = Lexer::new(&code);
            let mut tokens = Vec::new();
            while let Some(token) = lexer.next_token() {
                tokens.push(token);
            }
            tokens.push(Token::EOF);
            
            let mut parser = Parser::new(tokens);
            match parser.parse_stmts() {
                Ok(stmts) => {
                    let mut interpreter = Interpreter::new(env);
                    match interpreter.eval_stmts(&stmts) {
                        Ok(_) => Ok(()),
                        Err(e) if e == "return" || e == "break" || e == "continue" => Ok(()),
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(format!("Parse error: {}", e)),
            }
        }
        "shell" | "shell-like" => {
            eval_shell(&code, &mut env)
        }
        "javascript" | "javascript-like" | "lua" | "lua-like" => {
            // For JS/Lua, we'll use the Python-like interpreter as a fallback
            // with adjusted syntax handling (simplified)
            let mut lexer = Lexer::new(&code);
            let mut tokens = Vec::new();
            while let Some(token) = lexer.next_token() {
                tokens.push(token);
            }
            tokens.push(Token::EOF);
            
            let mut parser = Parser::new(tokens);
            match parser.parse_stmts() {
                Ok(stmts) => {
                    let mut interpreter = Interpreter::new(env);
                    match interpreter.eval_stmts(&stmts) {
                        Ok(_) => Ok(()),
                        Err(e) if e == "return" || e == "break" || e == "continue" => Ok(()),
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(format!("Parse error: {}", e)),
            }
        }
        _ => Err(format!("Unsupported language: {}", language)),
    };
    
    let end_time = unsafe { nexus_now_ms() };    let execution_time = (end_time - start_time) as u32;
    
    // Build output JSON
    let mut output = String::from("{\"stdout\":\"");
    output.push_str(&escape_json_string(&env.stdout));
    output.push_str("\",\"stderr\":\"");
    output.push_str(&escape_json_string(&env.stderr));
    output.push_str("\",\"exit_code\":");
    output.push_str(&env.exit_code.to_string());
    output.push_str(",\"execution_time_ms\":");
    output.push_str(&execution_time.to_string());
    
    // Add variables (for debugging)
    output.push_str(",\"variables\":{");
    let vars = env.get_all_vars();
    for (i, (k, v)) in vars.iter().enumerate() {
        if i > 0 { output.push(','); }
        output.push('"');
        output.push_str(&escape_json_string(k));
        output.push_str("\":\"");
        output.push_str(&escape_json_string(&format!("{}", v)));
        output.push('"');
    }
    output.push('}');
    
    output.push_str(",\"error\":");
    if let Err(e) = result {
        output.push('"');
        output.push_str(&escape_json_string(&e));
        output.push('"');
    } else {
        output.push_str("null");
    }
    output.push('}');
    
    // Write to output buffer
    let output_bytes = output.as_bytes();
    if output_bytes.len() > OUTPUT_BUFFER.len() {
        write_error("Output too large for buffer");
        return 1;
    }
    unsafe {
        ptr::copy_nonoverlapping(output_bytes.as_ptr(), OUTPUT_BUFFER.as_mut_ptr(), output_bytes.len());
        nexus_set_output_len(output_bytes.len() as u32);
    }
    
    log_info("Code execution completed");
    0 // Success
}
// =============================================================================
// Helper Functions
// =============================================================================

fn parse_json_string(json: &str, key: &str) -> Option<String> {
    let search_key = format!("\"{}\"", key);
    if let Some(key_pos) = json.find(&search_key) {
        if let Some(colon_pos) = json[key_pos..].find(':') {
            let start = key_pos + colon_pos + 1;
            let rest = &json[start..].trim_start();
            if rest.starts_with('"') {
                let mut end = 1;
                while end < rest.len() {
                    let ch = rest.as_bytes()[end];
                    if ch == b'"' {
                        return Some(rest[1..end].to_string());
                    } else if ch == b'\\' && end + 1 < rest.len() {
                        end += 2;
                    } else {
                        end += 1;
                    }
                }
            }
        }
    }
    None
}

fn parse_json_int(json: &str, key: &str, default: i32) -> i32 {
    let search_key = format!("\"{}\"", key);
    if let Some(key_pos) = json.find(&search_key) {
        if let Some(colon_pos) = json[key_pos..].find(':') {
            let start = key_pos + colon_pos + 1;
            let rest = &json[start..].trim_start();
            let mut end = 0;
            for (i, ch) in rest.chars().enumerate() {
                if ch.is_ascii_digit() || (i == 0 && ch == '-') {
                    end = i + 1;
                } else {
                    break;
                }
            }
            if end > 0 {
                return rest[..end].parse().unwrap_or(default);
            }
        }
    }
    default
}
fn escape_json_string(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() * 2);
    for ch in s.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                let code = ch as u32;
                escaped.push_str(&format!("\\u{:04x}", code));
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn write_error(msg: &str) {
    let error_json = format!(
        "{{\"stdout\":\"\",\"stderr\":\"\",\"exit_code\":1,\"execution_time_ms\":0,\"variables\":{{}},\"error\":\"{}\"}}",
        escape_json_string(msg)
    );
    let bytes = error_json.as_bytes();
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), OUTPUT_BUFFER.as_mut_ptr(), bytes.len());
        nexus_set_output_len(bytes.len() as u32);
    }
}

fn log_info(msg: &str) {
    unsafe {
        nexus_log(2, msg.as_ptr() as *const c_char, msg.len() as i32);
    }
}
