.PHONY: dev

dev:
    source venv/bin/activate
    uvicorn main:app --reload
