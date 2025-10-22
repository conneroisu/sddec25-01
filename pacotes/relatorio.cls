% ---
%
% Classe relatorio.cls para relatórios de disciplinas
%   by: Valdex de Jesus Santos <valdexsantos@ifba.edu.br>
%   Mais templates em  abre.ai/profwaldexsantos
%	
% ---

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pacotes/relatorio}[2024/05/18 v6.0] %nome da classe

% repassa os parametros ao abntex2
\DeclareOption*{
	\PassOptionsToClass{\CurrentOption}{abntex2}
}
%=====================================================
\ProcessOptions\relax % Processa as opções fornecidas à classe
\LoadClass{abntex2} % Carrega a classe abntex2 como base
%===============================================

\PassOptionsToClass{
	12pt,				% tamanho da fonte
	a4paper,			% tamanho do papel.
	sumario=tradicional,		% seleciona o sumário tradicional
	}{abntex2} %chama a classe ABNT
 
%===================%===================%

% Pacotes utilizados
\usepackage{times}   % Usa a fonte Times
\usepackage[utf8]{inputenc}   % Codificacao do documento (conversão automática dos acentos)
% babel is loaded by abntex2, so we don't need to load it here
\usepackage[T1]{fontenc}   % Selecao de codigos de fonte
\usepackage{graphicx}   % Inclusão de gráficos
\usepackage{algorithm2e}   % Para trabalhar com algoritmos
\usepackage{amsfonts,amsthm, amssymb, amsmath}   % Fonte e símbolos matemáticos
\usepackage{caption}   % Legendas
\usepackage{subcaption}   % Sublegendas
\usepackage{hyperref}   % Links
%\usepackage{nameref}   % Referências por nome
\usepackage{etoolbox}   % Ferramentas de patching
\usepackage{float}	%PERMITE usar H em figura
\usepackage{cancel}     %cancelar termos numa expressão matemática
\usepackage{verbatim}			% texto é interpretado como escrito no documento
\usepackage{multirow, array}	% múltiplas linhas e colunas em     tabelas
\usepackage{indentfirst}	% indenta o primeiro parágrafo de cada seção.
\usepackage{xcolor, colortbl}			% comandos de cores
\usepackage{pdflscape}   % permite colocar página em formato paisagem
\usepackage{lipsum}				% para geração de dummy text
\usepackage{setspace}  % Pacote para controle de espaçamento
\usepackage{multicol} %MULTICOLUNAS
\usepackage{enumerate}

%===========%===================%===================%

% Using traditional bibliography approach for compatibility
\usepackage[numbers,sort&compress]{natbib}
\bibliographystyle{IEEEtran}
% Alternative: use biblatex (commented out for compatibility)
% \usepackage{csquotes}
% \usepackage[style=numeric,sorting=none,backend=biber]{biblatex}

 %===========%===================%===================%

% Configurações do hyperref
\hypersetup{
     	%pagebackref=true,
		pdftitle={\@title}, 
		pdfauthor={\@author},
    	pdfsubject={\imprimirpreambulo},
	    pdfcreator={Valdex Santos},
		colorlinks=true,       		% false: boxed links; true: colored links
    	linkcolor=black,          	% cor dos links internos
    	citecolor=black,        		% cor dos links das referências.
    	filecolor=magenta,      		% cor dos links de arquivos
		urlcolor=blue,               %cor das urls
		bookmarksdepth=4
}

%===========%===================%===================%
%DEFININDO COMANDOS PERSONALIZADOS
% Comandos personalizados para a capa e folha de rosto
\newcommand{\logoInst}[1]{
\newcommand{\logo}{#1}
}

\newcommand{\nomeInst}[1]{
\newcommand{\inst}{#1}
}

\newcommand{\nomeCurso}[1]{
\newcommand{\curso}{#1}
}

%==============================================

\renewcommand{\imprimircapa}{%
    \begin{capa}%
        % Seu conteúdo da capa aqui...
    \end{capa}
}


% redefinição da cap
\renewcommand{\imprimircapa}{%

 \begin{capa}%
  
     \hspace{-1.0cm}
		\begin{minipage}{.2\textwidth}
			\logo
		\end{minipage}\hspace{-1.0cm}
		\begin{minipage}{.8\textwidth}
			\begin{center}
    \ABNTEXchapterfont\large{\inst\\ \curso}
    \end{center}
		\end{minipage}
    
    \vfill
    \begin{center}
    \MakeUppercase{\ABNTEXchapterfont\bfseries\Large\imprimirtitulo}
      \vskip 5mm    
    
    \normalfont\imprimirautor
    \vfill
    
    \large\imprimirlocal

    \large\imprimirdata
    
    \vspace*{1cm}
    \end{center}
  \end{capa}
}


% Redefinição da folha de rosto
\makeatletter
\renewcommand{\folhaderostocontent}{
\begin{center}
{\ABNTEXchapterfont\large \inst\\ \curso}
    
    \vspace*{3.5cm}
    
\MakeUppercase{\ABNTEXchapterfont\bfseries\Large\imprimirtitulo}


\end{center}
\begin{center}
{\ABNTEXchapterfont\large\imprimirautor}
\end{center}

\vspace{1cm}

\abntex@ifnotempty{\imprimirpreambulo}{%
  \hspace{.45\textwidth}
  \begin{minipage}{.5\textwidth}
  \SingleSpacing
  \imprimirpreambulo
  \end{minipage}%
  \vspace*{\fill}
}%


{\large\imprimirorientadorRotulo~\imprimirorientador\par}

\abntex@ifnotempty{\imprimircoorientador}{%
  {\noindent\large\imprimircoorientadorRotulo~\imprimircoorientador}%
}%

\vspace*{\fill}
\begin{center}
{\large\imprimirlocal}

\par

{\large\imprimirdata}
\vspace*{1cm}
\end{center}
}


%================================================
%NOVOS COMANDO
\newcommand{\aspas}[1]{``#1''} %comando para aspas
%\newcommand{\citeonline}[2][]{\textcite[#1]{#2}} %comando alternativo para citações usando \citeonline em vez de \textcite
