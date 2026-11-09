//! The `CubeCL` wordmark, drawn in braille: `cubecl-wordmark-hero.svg` from the
//! Tracel website, rasterized and thresholded to 6 rows of 2×4 dots per
//! cell. The mark and the letters are kept apart so each takes its own colour,
//! as the site's accented heading does.

/// The mark, one line per row.
pub(crate) const MARK: [&str; 6] = [
    "⢀⣀⡀⣠⣶⣿⣿⣿⣷⣤⡀",
    "⣿⡟⣼⣿⣿⡿⣛⡻⣿⣿⣷",
    "⠈⢸⣿⣿⡏⠈⠉⠉⢉⠉⠁",
    "⠀⠘⣿⣿⣿⣶⠟⣹⣿⣿⡆",
    "⠀⠀⠈⠙⠛⣫⣾⣿⣿⠟⠁",
    "⠀⠀⠀⠀⠀⢿⣿⡿⠋⠀⠀",
];

/// The letters, one line per row, each padded to the same width.
pub(crate) const LETTERS: [&str; 6] = [
    "⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣤⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⣄⣀⠀⠀⠀⢀⣀⡀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⢀⣾⣿⠿⠛⠻⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠟⠻⢿⣿⣆⠀⢸⣿⡇⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⣼⣿⡏⠀⠀⠀⠈⠉⠉⠀⣶⣶⠀⠀⣶⣶⠀⢸⣿⣷⣾⣿⣷⣦⠀⢀⣴⣿⢿⣷⣦⠀⠀⠀⠀⢸⣿⡏⠀⠀⠀⠀⠉⠉⠀⢸⣿⡇⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⢹⣿⣇⠀⠀⠀⢀⣤⣤⠀⣿⣿⠀⠀⣿⣿⠀⢸⣿⡏⠀⠀⢹⣿⡇⣿⣿⣶⣶⣾⣿⡇⠀⠀⠀⢸⣿⣧⠀⠀⠀⠀⣤⣤⠀⢸⣿⡇⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠈⠻⣿⣷⣶⣶⣿⡿⠃⠀⢻⣿⣦⣴⣿⡿⠀⢸⣿⣿⣦⣴⣿⡿⠁⠹⣿⣦⣤⣴⡶⠂⠀⠀⠀⠀⠻⣿⣷⣶⣶⣾⡿⠋⠀⢸⣿⣷⣶⣶⣶",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠁⠀⠀⠀⠀⠈⠉⠉⠁⠀⠀⠈⠉⠁⠉⠉⠉⠀⠀⠀⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠀⠀⠀⠈⠉⠉⠉⠉⠉",
];
