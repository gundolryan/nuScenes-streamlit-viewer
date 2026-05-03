# nuScenes-streamlit-viewer
nuScenes 씬을 6개 카메라(2x3)로 동시에 확인할 수 있는 Streamlit 기반 뷰어입니다.

## 주요 기능
- Scene 검색 및 선택
- 6카메라 동시 재생
- 재생 컨트롤: `Prev / Play-Pause / Next`
- FPS 조절
- Frame / Sample 슬라이더 이동
- Scene 메타데이터 토글 조회
- 선택 Scene 기준 앞/뒤 Scene 백그라운드 프리로드

## 요구 환경
- Python 3.10+
- Streamlit
- OpenCV (`opencv-python`)
- nuScenes devkit (`nuscenes-devkit`)

## 실행 방법
```bash
cd <your-repo-root>
conda activate <your-conda-env>
streamlit run tools/nuscenes_streamlit_viewer.py --logger.level=error
```

브라우저 접속:
- `http://localhost:8501`

## 사용자별 데이터셋 경로 설정
앱 실행 후 좌측 사이드바의 `Dataset Path`에 각자 로컬 경로를 입력하면 됩니다.

예시:
- Linux: `/data/nuscenes`
- Windows: `D:\\datasets\\nuscenes`

## 참고
- 이 뷰어는 mp4 파일 저장 없이 브라우저에서 직접 재생합니다.
- 첫 로딩 시에는 선택된 Scene 프레임 버퍼를 구성하므로 잠시 시간이 걸릴 수 있습니다.

## 설치 예시 (선택)
환경마다 패키지 구성이 다를 수 있으므로, 아래 명령어로 설치하세요.

```bash
pip install streamlit opencv-python nuscenes-devkit numpy
```
