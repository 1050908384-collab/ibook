from typing import Dict, Any, Optional
from fastapi import APIRouter


router = APIRouter()


class HumanIntervention:
    def __init__(self):
        self.intervention_requests: Dict[str, Dict[str, Any]] = {}
    
    def request_intervention(self, session_id: str, reason: str, context: Dict[str, Any]) -> str:
        request_id = f"intervention_{session_id}_{len(self.intervention_requests) + 1}"
        self.intervention_requests[request_id] = {
            "session_id": session_id,
            "reason": reason,
            "context": context,
            "status": "pending",
            "decision": None,
            "timestamp": None
        }
        return request_id
    
    def get_intervention_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self.intervention_requests.get(request_id)
    
    def submit_decision(self, request_id: str, decision: str, notes: str = "") -> bool:
        if request_id not in self.intervention_requests:
            return False
        
        self.intervention_requests[request_id].update({
            "status": "resolved",
            "decision": decision,
            "notes": notes
        })
        return True
    
    def list_pending_requests(self) -> list:
        return [
            req for req in self.intervention_requests.values() 
            if req["status"] == "pending"
        ]


@router.post("/request")
def request_intervention(session_id: str, reason: str, context: Dict[str, Any]):
    hi = HumanIntervention()
    request_id = hi.request_intervention(session_id, reason, context)
    return {"request_id": request_id, "status": "pending"}


@router.get("/{request_id}")
def get_intervention(request_id: str):
    hi = HumanIntervention()
    return hi.get_intervention_request(request_id)


@router.post("/{request_id}/decision")
def submit_decision(request_id: str, decision: str, notes: str = ""):
    hi = HumanIntervention()
    success = hi.submit_decision(request_id, decision, notes)
    return {"success": success}