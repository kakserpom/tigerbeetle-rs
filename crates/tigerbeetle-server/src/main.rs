//! `tigerbeetle` binary entry point (port of `src/tigerbeetle/main.zig`).
//!
//! Dispatches the parsed [`cli::Command`] and applies upstream's exit-code contract
//! (`stdx/flags.zig` `fatal`: `error: {message}` on stderr, exit 1; `-h`/`--help`: usage on
//! stdout, exit 0).

mod cli;

use cli::ParseFailure;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match cli::parse_args(&argv) {
        Ok(command) => match cli::run_command(&command) {
            Ok(()) => {}
            Err(message) => fatal(&message),
        },
        Err(ParseFailure::Help(help)) => {
            println!("{help}");
        }
        Err(ParseFailure::Fatal(message)) => fatal(&message),
    }
}

/// Upstream `stdx/flags.zig` `fatal`: print `error: {message}` to stderr and exit 1.
fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}
