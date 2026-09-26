# Движок — lualatex: fontspec и polyglossia работают только под ним, и только
# он вставляет шрифты с кириллицей из системы.
$pdf_mode = 4;
$lualatex = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';
$clean_ext = 'synctex.gz run.xml bbl';
