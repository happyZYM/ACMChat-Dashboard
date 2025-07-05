from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
import os
from loguru import logger
import sys
import uvicorn
import json
import aiosqlite
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pydantic import BaseModel
import secrets

listen_port = os.getenv("LISTEN_PORT", "8000")
listen_host = os.getenv("LISTEN_HOST", "0.0.0.0")
config_path = os.getenv("CONFIG_PATH", "./config.json")
database_path = os.getenv("DATABASE_PATH", "./data.sqlite3")
logging_level = os.getenv("LOGGING_LEVEL", "TRACE")

# Configure loguru properly - remove default handler and add new one with correct level
logger.remove()  # Remove the default handler
logger.add(sys.stderr, level=logging_level)  # Add new handler with specified level

security_bearer = HTTPBearer()
security_basic = HTTPBasic()

class APICallRecord(BaseModel):
    timestamp: int
    model_id: str
    user_email: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

def normalize_path(path: str) -> str:
    """
    处理路径，支持绝对路径、用户目录和相对路径

    Args:
        path: 输入路径

    Returns:
        处理后的路径
    """
    if path.startswith("/") or path.startswith("~"):
        # 绝对路径，如果是~开头则展开为用户目录
        return os.path.expanduser(path)
    elif path.startswith("./"):
        # 相对于脚本本身的相对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, path[2:])  # 去掉'./'前缀
    else:
        # 其他情况保持原样（相对于pwd的相对路径）
        return path

# 处理config_path和database_path路径
config_path = normalize_path(config_path)
database_path = normalize_path(database_path)

with open(config_path, "r") as f:
    config = json.load(f)

admin_api_key = config.get("admin_api_key", "")
report_api_key = config.get("report_api_key", "")
session_secret_key = config.get("session_secret_key", secrets.token_hex(32))

async def init_database():
    """Initialize the database and create tables if they don't exist"""
    async with aiosqlite.connect(database_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                user_email TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        logger.info("Database initialized successfully")

async def verify_admin_session(request: Request):
    """Verify admin session for dashboard access"""
    if not request.session.get("authenticated"):
        raise HTTPException(
            status_code=302,
            headers={"Location": "/login"}
        )
    return True

async def verify_report_key(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    """Verify report API key for API access using Bearer authentication"""
    if credentials.credentials != report_api_key:
        raise HTTPException(status_code=403, detail="Invalid report API key")
    return credentials.credentials

async def get_24h_stats():
    """Get statistics for the past 24 hours"""
    twenty_four_hours_ago = int(time.time()) - (24 * 60 * 60)
    
    async with aiosqlite.connect(database_path) as db:
        # Total chats (assuming each API call is a chat)
        cursor = await db.execute(
            "SELECT COUNT(*) FROM api_calls WHERE timestamp >= ?",
            (twenty_four_hours_ago,)
        )
        total_chats = (await cursor.fetchone())[0]
        
        # Total tokens
        cursor = await db.execute(
            "SELECT SUM(input_tokens + output_tokens) FROM api_calls WHERE timestamp >= ?",
            (twenty_four_hours_ago,)
        )
        result = await cursor.fetchone()
        total_tokens = result[0] if result[0] is not None else 0
        
        # Total cost
        cursor = await db.execute(
            "SELECT SUM(cost_usd) FROM api_calls WHERE timestamp >= ?",
            (twenty_four_hours_ago,)
        )
        result = await cursor.fetchone()
        total_cost = result[0] if result[0] is not None else 0.0
        
        return {
            "total_chats": total_chats,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6)
        }

async def get_recent_logs(page: int = 1, page_size: int = 100):
    """Get recent API call logs with pagination"""
    offset = (page - 1) * page_size
    
    async with aiosqlite.connect(database_path) as db:
        # Get total count for pagination
        cursor = await db.execute("SELECT COUNT(*) FROM api_calls")
        total_count = (await cursor.fetchone())[0]
        
        # Get paginated results
        cursor = await db.execute("""
            SELECT timestamp, model_id, user_email, input_tokens, output_tokens, cost_usd
            FROM api_calls
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (page_size, offset))
        
        logs = []
        async for row in cursor:
            logs.append({
                "timestamp": row[0],
                "model_id": row[1],
                "user_email": row[2],
                "input_tokens": row[3],
                "output_tokens": row[4],
                "total_tokens": row[3] + row[4],
                "cost_usd": round(row[5], 6)
            })
        
        total_pages = (total_count + page_size - 1) // page_size  # Ceiling division
        
        return {
            "logs": logs,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_count": total_count,
                "page_size": page_size,
                "has_prev": page > 1,
                "has_next": page < total_pages
            }
        }

async def get_users_report():
    """Get user consumption statistics"""
    async with aiosqlite.connect(database_path) as db:
        cursor = await db.execute("""
            SELECT 
                user_email,
                COUNT(*) as total_calls,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(input_tokens + output_tokens) as total_tokens,
                SUM(cost_usd) as total_cost
            FROM api_calls
            GROUP BY user_email
            ORDER BY total_cost DESC
        """)
        
        users = []
        async for row in cursor:
            users.append({
                "email": row[0],
                "total_calls": row[1],
                "total_input_tokens": row[2],
                "total_output_tokens": row[3],
                "total_tokens": row[4],
                "total_cost": round(row[5], 6)
            })
        
        return users

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    await init_database()
    logger.info("Application started")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

app = FastAPI(lifespan=lifespan)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=session_secret_key)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Routes
@app.get("/")
async def root(request: Request):
    """Redirect to dashboard or login"""
    if request.session.get("authenticated"):
        return RedirectResponse(url="/dashboard", status_code=302)
    else:
        return RedirectResponse(url="/login", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, _: bool = Depends(verify_admin_session)):
    """Main dashboard page"""
    stats = await get_24h_stats()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats
    })

@app.get("/dashboard/logs", response_class=HTMLResponse)
async def dashboard_logs(request: Request, page: int = 1, _: bool = Depends(verify_admin_session)):
    """Dashboard logs page"""
    if page < 1:
        page = 1
    
    result = await get_recent_logs(page=page, page_size=100)
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "logs": result["logs"],
        "pagination": result["pagination"]
    })

@app.get("/dashboard/users_report", response_class=HTMLResponse)
async def dashboard_users_report(request: Request, _: bool = Depends(verify_admin_session)):
    """Dashboard users report page"""
    users = await get_users_report()
    return templates.TemplateResponse("users_report.html", {
        "request": request,
        "users": users
    })

@app.post("/api/record_api_call")
async def record_api_call(
    record: APICallRecord,
    _: str = Depends(verify_report_key)
):
    """Record an API call"""
    try:
        async with aiosqlite.connect(database_path) as db:
            await db.execute("""
                INSERT INTO api_calls (timestamp, model_id, user_email, input_tokens, output_tokens, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp,
                record.model_id,
                record.user_email,
                record.input_tokens,
                record.output_tokens,
                record.cost_usd
            ))
            await db.commit()
        
        logger.info(f"Recorded API call: {record.user_email} - {record.model_id} - ${record.cost_usd}")
        return {"status": "success", "message": "API call recorded successfully"}
    
    except Exception as e:
        logger.error(f"Error recording API call: {e}")
        raise HTTPException(status_code=500, detail="Failed to record API call")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    """Handle login form submission"""
    if secrets.compare_digest(password, admin_api_key):
        request.session["authenticated"] = True
        return RedirectResponse(url="/dashboard", status_code=302)
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid password"
        })

@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session"""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)

if __name__ == "__main__":
    uvicorn.run(app, host=listen_host, port=int(listen_port))
