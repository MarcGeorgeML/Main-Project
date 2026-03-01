from pydantic import BaseModel

# {
#     'request_id': '4b1268e7-e2b2-4d74-9684-89d84a1453a3', 
#     'data': {
#         "user_id": "779d5f22-ef1a-46fc-ad0f-c1fe9ec470ba", 
#         "type": "text", 
#         "message": "hello"
#     }
# }


class RequestData(BaseModel):
    user_id: str
    type: str
    data: str
    latest_emotion: str | None = None


class RequestType(BaseModel):
    request_id: str
    data: RequestData 