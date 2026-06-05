use core::iter::Iterator;

use libafl::mutators::Tokens;

const DICTIONARY: &'static str = include_str!("dictionary.txt");

/// Returns an iterator over all the strings in the dictionary.
pub fn strings() -> impl Iterator<Item = &'static str> {
    DICTIONARY.lines()
}

/// Builds a dictionary of interesting tokens for use in fuzzing.
pub fn tokens() -> Tokens {
    let mut tokens = Tokens::new();
    tokens.add_tokens(strings().map(|s| s.as_bytes().to_vec()));
    assert!(!tokens.is_empty());
    tokens
}
