import sys
from pathlib import Path
current_directory = Path(__file__).parent.resolve()
project_top_level_directory = current_directory.parent
sys.path.append(str(project_top_level_directory))
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, APIRouter, BackgroundTasks, Depends
import uvicorn
import gradio as gr 

from utils.message import ChatMessage
from utils.schema import TextContent
from service.payload import *

from agent.brain import MasterBrain
from service.webui import webui

#设置根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

rotating_handler = RotatingFileHandler('app.log')
rotating_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(rotating_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(stream_handler)
    

master_brain = MasterBrain.initialize()
app = FastAPI()
router = APIRouter()

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    root_logger.error(f"An error occurred: {exc}")
    return {"error": str(exc)}, 500


from store import AICChatStoreLocal

store = AICChatStoreLocal()


@router.post("/api/chat_list", response_model=GetChatListResponse)
async def getChatList(req: GetChatListRequest):
    return GetChatListResponse(data=store.get_chat_list(include=req.include))


@router.post("/api/chat/create", response_model=CreateChatResponse)
async def createChat(req: CreateChatRequest):
    return CreateChatResponse(data=store.create_chat(member=req.member))


@router.get("/api/chat", response_model=GetChatResponse)
async def getChat(req: GetChatRequest = Depends(GetChatRequest)):
    return GetChatResponse(data=store.get_chat(chat_id=req.chat_id))


@router.post("/api/chat/update", response_model=UpdateChatResponse)
async def updateChat(req: UpdateChatRequest):
    return UpdateChatResponse(data=store.update_chat(chat=req.chat))


@router.post("/api/chat", response_model=ChatEventResponse)
async def chatSync(req: ChatEventRequest):
    """一问一答"""
    receiver = req.message.receiver
    sender = req.message.sender
    sender_detail = store.get_identity(sender.id)
    kwargs = {"exclude_items": ["regular_schedule", "story"]}
    reply, thought = await master_brain.handle_interaction(
        receiver.id,
        [sender_detail.name],
        sender_detail.name,
        req.message.content.content,
        verbose=True,
        **kwargs,
    )
    response = ChatEventResponse(
        data=ChatMessage(
            sender=receiver,
            receiver=sender,
            content=TextContent(content=reply),
            context=req.message.context,
            metadata={"thought": thought},
        )
    )
    # 更新事件记录、ChatSession
    store.add_event(req.message)
    store.add_event(response.data)
    chat = store.get_chat(chat_id=req.message.context.chat_id)
    chat.event_ids.extend([req.message.event_id, response.data.event_id])
    store.update_chat(chat=chat)
    return response

ui = webui()
app = gr.mount_gradio_app(app, ui, path="/ai-companion/api/gradio")
app.include_router(router, prefix="/ai-companion")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level='debug')
