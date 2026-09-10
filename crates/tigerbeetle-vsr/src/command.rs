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

impl core::fmt::Display for Command {
    /// Port of upstream's `{any}` format for `vsr.Command` enum: snake_case tag name.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Reserved => f.write_str("reserved"),
            Self::Ping => f.write_str("ping"),
            Self::Pong => f.write_str("pong"),
            Self::PingClient => f.write_str("ping_client"),
            Self::PongClient => f.write_str("pong_client"),
            Self::Request => f.write_str("request"),
            Self::Prepare => f.write_str("prepare"),
            Self::PrepareOk => f.write_str("prepare_ok"),
            Self::Reply => f.write_str("reply"),
            Self::Commit => f.write_str("commit"),
            Self::ExitView => f.write_str("exit_view"),
            Self::JoinView => f.write_str("join_view"),
            Self::GetView => f.write_str("get_view"),
            Self::GetHeaders => f.write_str("get_headers"),
            Self::GetPrepare => f.write_str("get_prepare"),
            Self::GetReply => f.write_str("get_reply"),
            Self::Headers => f.write_str("headers"),
            Self::Eviction => f.write_str("eviction"),
            Self::GetBlocks => f.write_str("get_blocks"),
            Self::Block => f.write_str("block"),
            Self::View => f.write_str("view"),
            Self::Deprecated12 => f.write_str("deprecated_12"),
            Self::Deprecated21 => f.write_str("deprecated_21"),
            Self::Deprecated22 => f.write_str("deprecated_22"),
            Self::Deprecated23 => f.write_str("deprecated_23"),
        }
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
