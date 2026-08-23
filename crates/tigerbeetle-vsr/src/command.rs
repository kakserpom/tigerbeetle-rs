//! Port of `vsr.Command` from `src/vsr.zig`.

/// The Viewstamped Replication protocol command for a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Command {
    Reserved = 0,

    Ping = 1,
    Pong = 2,

    PingClient = 3,
    PongClient = 4,

    Request = 5,
    Prepare = 6,
    PrepareOk = 7,
    Reply = 8,
    Commit = 9,

    ExitView = 10,
    JoinView = 11,
    GetView = 13,

    GetHeaders = 14,
    GetPrepare = 15,
    GetReply = 16,
    GetBlocks = 19,

    Headers = 17,

    Eviction = 18,

    Block = 20,

    View = 24,

    // If a command is removed from the protocol, its ordinal is added here and can't be re-used.
    Deprecated12 = 12, // .view without checkpoint
    Deprecated21 = 21, // .request_sync_checkpoint
    Deprecated22 = 22, // .sync_checkpoint
    Deprecated23 = 23, // .view with an older version of CheckpointState
}

impl Command {
    /// Upstream's comptime assertion that ordinals are dense (`@intFromEnum(command) <
    /// values(Command).len` for every command) pins the enum to exactly these 24 variants:
    const ALL: [Command; 25] = [
        Command::Reserved,
        Command::Ping,
        Command::Pong,
        Command::PingClient,
        Command::PongClient,
        Command::Request,
        Command::Prepare,
        Command::PrepareOk,
        Command::Reply,
        Command::Commit,
        Command::ExitView,
        Command::JoinView,
        Command::Deprecated12,
        Command::GetView,
        Command::GetHeaders,
        Command::GetPrepare,
        Command::GetReply,
        Command::Headers,
        Command::Eviction,
        Command::GetBlocks,
        Command::Block,
        Command::Deprecated21,
        Command::Deprecated22,
        Command::Deprecated23,
        Command::View,
    ];

    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        Self::ALL.iter().copied().find(|command| *command as u8 == value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors upstream's comptime assertion that command ordinals are dense:
    /// `@intFromEnum(command) < values(Command).len` for every command.
    #[allow(clippy::cast_possible_truncation)]
    #[test]
    fn ordinals_are_dense() {
        for (index, command) in Command::ALL.iter().enumerate() {
            assert_eq!(*command as u8, index as u8);
            assert_eq!(Command::from_u8(index as u8), Some(*command));
        }
        assert_eq!(Command::ALL.len(), 25);
        // Ordinals past the last command stay unused:
        assert_eq!(Command::from_u8(25), None);
        assert_eq!(Command::from_u8(u8::MAX), None);
    }
}
