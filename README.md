# Node2Vec

> [Elior Cohen의 node2vec](https://github.com/eliorc/node2vec)을 uv 및 몇가지 부족한 부분을 보완한 저장소입니다.

## 개발 환경 설정

```bash
# 의존성 설치
uv sync

# 개발 의존성 포함 설치
uv sync --extra dev

# 테스트 실행
uv run pytest

# 커버리지 포함 테스트 실행
uv run pytest --cov=node2vec --cov-report=html
```
