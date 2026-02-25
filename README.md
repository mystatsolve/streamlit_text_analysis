# 텍스트 빈도 분석 도구

CSV 파일을 업로드하여 한국어 텍스트 빈도 분석과 워드 클라우드를 생성하는 정적 웹앱입니다.

## 기능

- **CSV 업로드** — `sentence`(또는 `abstract`) · `year` 컬럼 자동 감지
- **불용어 관리** — 기본 불용어 제공, 사용자 추가/삭제 가능
- **연도 필터** — 특정 연도 선택 분석
- **빈도 분석표** — 단어·빈도·비율, 정렬·검색·페이지 이동
- **워드 클라우드** — 색상 테마 선택, PNG 저장
- **CSV 다운로드** — 분석 결과 저장

## GitHub Pages 배포 방법

### 1단계 — GitHub 저장소 생성

1. [github.com](https://github.com) 로그인
2. 우측 상단 **+** → **New repository**
3. Repository name: `text-freq-analysis` (원하는 이름)
4. Public 선택 → **Create repository**

### 2단계 — 파일 업로드

```bash
# 방법 A: git 명령어 사용
git init
git add index.html README.md
git commit -m "Add text frequency analysis web app"
git branch -M main
git remote add origin https://github.com/사용자이름/text-freq-analysis.git
git push -u origin main
```

또는 GitHub 웹에서 **Add file → Upload files** 로 `index.html` 업로드.

### 3단계 — GitHub Pages 활성화

1. 저장소 → **Settings** 탭
2. 왼쪽 메뉴 **Pages**
3. Source: **Deploy from a branch** → Branch: `main` / `/ (root)` → **Save**
4. 약 1~2분 후 `https://사용자이름.github.io/text-freq-analysis/` 접속 가능

## CSV 형식

| sentence | year |
|---|---|
| 연구 내용 텍스트... | 2022 |
| 분석 결과... | 2023 |

- `sentence` 대신 `abstract`, `text`, `내용` 등도 자동 인식
- UTF-8 또는 UTF-8 BOM 인코딩 권장
