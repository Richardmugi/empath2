from fastapi import FastAPI
from app.main import router
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()

# Mount all routes from main.py
app.include_router(router)

# Enable session management (for login)
app.add_middleware(SessionMiddleware, secret_key="supersecret")  

# Re-export the app for deployment tools
application = app
