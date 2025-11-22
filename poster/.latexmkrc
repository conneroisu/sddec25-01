# Latexmk configuration for poster compilation
# Work around tikzposter v2.0 "Missing character: 1=1 in nullfont" bug

# Force completion despite tikzposter's spurious "missing character" warnings
# The PDF generates correctly - these warnings are a known tikzposter bug
$force_mode = 1;
