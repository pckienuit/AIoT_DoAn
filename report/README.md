# Report LaTeX Template

Thu muc nay chua ban bao cao LaTeX cho do an AIoT Flight Face Lookup.

## Bien dich

Mau bao cao hien dung pdfLaTeX voi `extarticle`, `T5`, `inputenc` va `lmodern`, giong form mon hoc ban dua:

```powershell
cd report
latexmk -pdf main.tex
```

Neu khong co `latexmk`, chay thu cong:

```powershell
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

File PDF dau ra: `report/main.pdf`.

## Cau truc de xuat

- `main.tex`: file goc, khai bao goi va include cac chuong.
- `chapters/00_frontmatter.tex`: trang bia, loi cam on, tom tat.
- `chapters/01_introduction.tex`: gioi thieu de tai, muc tieu, pham vi.
- `chapters/02_background_requirements.tex`: co so ly thuyet va yeu cau he thong.
- `chapters/03_system_architecture.tex`: kien truc tong the Web - Server - Edge.
- `chapters/04_design_implementation.tex`: thiet ke va hien thuc tung khoi.
- `chapters/05_evaluation.tex`: kiem thu, benchmark, danh gia.
- `chapters/06_conclusion.tex`: ket luan va huong phat trien.
- `chapters/appendix.tex`: phu luc cai dat, API, cau hinh.
- `references.bib`: tai lieu tham khao.
- `figures/`: dat anh minh hoa, so do, screenshot demo.

## Viec can bo sung

- Thong tin truong, khoa, mon hoc, giang vien, nhom sinh vien.
- Anh giao dien web, anh MaixCAM, anh demo luong nhan dien.
- Bang so lieu benchmark cuoi cung neu thay doi.
- Danh sach tai lieu tham khao dung theo yeu cau cua mon hoc.
