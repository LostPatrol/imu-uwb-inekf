<!-- Manuscript entry points and compilation requirements. -->
# 论文与数学说明

当前研究材料：

- [英文论文](recovery_bootstrap_paper.pdf) 与 [TeX 源码](recovery_bootstrap_paper.tex)。
- [中文数学说明](recovery_bootstrap_math_zh.pdf) 与 [TeX 源码](recovery_bootstrap_math_zh.tex)。
- `recovery_bootstrap_refs.bib`：参考文献。
- `recovery_bootstrap_figures`：当前论文配图。

其他 TeX/PDF 是算法修正、加速、增广方法与共同协议的历史说明，保留供比较；
当前结论以 [研究状态](../docs/research-status.md) 为准。

## 编译

Windows 安装含 IEEEtran、ctex 的 TeX Live 后，从仓库根运行：

```powershell
& ./tex/build_recovery_paper.ps1 -TexBin 'D:/Texlive/2026/bin/windows'
```

将 `-TexBin` 改为本机编译器目录。英文稿使用 pdfLaTeX/BibTeX，中文稿使用 XeLaTeX。
中间文件写入忽略的过程目录，最终 PDF 位于本目录。

历史 TeX 引用的项目内图片也已保留；请在 `tex` 目录下编译这些文档，使相对路径有效。
