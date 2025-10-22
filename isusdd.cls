%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iowa State University Senior Design Document Class
% Modern English LaTeX Document Class
% Based on standard report class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{isusdd}[2025/10/22 ISU Senior Design Document Class]

% Load base class
\LoadClass[12pt,oneside,letterpaper]{report}

% Required packages
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[english]{babel}
\RequirePackage{csquotes}
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{tocloft}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{fancyhdr}
\RequirePackage{hyperref}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{listings}
\RequirePackage{xcolor}
\RequirePackage{caption}
\RequirePackage{subcaption}

% Page geometry
\geometry{
    letterpaper,
    top=1in,
    bottom=1in,
    left=1in,
    right=1in,
    headheight=15pt
}

% Line spacing
\onehalfspacing

% Define document metadata commands
\newcommand{\@institution}{}
\newcommand{\institution}[1]{\renewcommand{\@institution}{#1}}

\newcommand{\@department}{}
\newcommand{\department}[1]{\renewcommand{\@department}{#1}}

\newcommand{\@course}{}
\newcommand{\course}[1]{\renewcommand{\@course}{#1}}

\newcommand{\@semester}{}
\newcommand{\semester}[1]{\renewcommand{\@semester}{#1}}

\newcommand{\@teamname}{}
\newcommand{\teamname}[1]{\renewcommand{\@teamname}{#1}}

\newcommand{\@teammembers}{}
\newcommand{\teammembers}[1]{\renewcommand{\@teammembers}{#1}}

\newcommand{\@documenttype}{}
\newcommand{\documenttype}[1]{\renewcommand{\@documenttype}{#1}}

\newcommand{\@location}{}
\newcommand{\location}[1]{\renewcommand{\@location}{#1}}

\newcommand{\@projectdescription}{}
\newcommand{\projectdescription}[1]{\renewcommand{\@projectdescription}{#1}}

% Custom title page
\renewcommand{\maketitle}{
    \begin{titlepage}
        \centering
        \vspace*{1cm}

        {\Large\textbf{\@institution}\\}
        \vspace{0.5cm}
        {\large\@department\\}
        \vspace{1.5cm}

        {\Large\textbf{\@course}\\}
        \vspace{2cm}

        {\huge\bfseries\@title\\}
        \vspace{2cm}

        {\Large\@teamname\\}
        \vspace{0.5cm}
        {\large\@teammembers\\}
        \vspace{2cm}

        \vfill

        {\large\@location\\}
        {\large\@semester\\}
        \vspace{1cm}

    \end{titlepage}
}

% Executive Summary environment (renamed from abstract to avoid conflict)
\renewenvironment{abstract}{
    \cleardoublepage
    \chapter*{Executive Summary}
    \addcontentsline{toc}{chapter}{Executive Summary}
}{
    \cleardoublepage
}

% Keywords command
\newcommand{\keywords}[1]{
    \vspace{1em}
    \noindent\textbf{Keywords}: #1
}

% Chapter formatting
\titleformat{\chapter}[display]
    {\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter}{20pt}{\Huge}
\titlespacing*{\chapter}{0pt}{50pt}{40pt}

% Section formatting
\titleformat{\section}
    {\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}
    {\normalfont\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
    {\normalfont\normalsize\bfseries}{\thesubsubsection}{1em}{}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Plain style for chapter pages
\fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[C]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
}

% Table of contents formatting
\renewcommand{\contentsname}{Table of Contents}
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

% Hyperref setup
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=blue,
    filecolor=magenta,
    urlcolor=blue,
    pdftitle={\@title},
    pdfauthor={\@teamname}
}

% Code listing style
\lstdefinestyle{codestyle}{
    backgroundcolor=\color{gray!10},
    commentstyle=\color{green!50!black},
    keywordstyle=\color{blue},
    numberstyle=\tiny\color{gray},
    stringstyle=\color{red!60!black},
    basicstyle=\ttfamily\small,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    frame=single
}
\lstset{style=codestyle}

% Custom commands for consistency
\newcommand{\requirements}{\chapter{Requirements and Constraints}}
\newcommand{\design}{\chapter{Design}}
\newcommand{\implementation}{\chapter{Implementation}}
\newcommand{\testing}{\chapter{Testing}}

\endinput
