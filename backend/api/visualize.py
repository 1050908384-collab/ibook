from fastapi import APIRouter, HTTPException
from orchestrator.harness import Harness

router = APIRouter()
harness = Harness()

@router.get("/profile/{session_id}")
async def get_profile_data(session_id: str):
    try:
        profile = harness.state_manager.get_profile(session_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stage/{session_id}")
async def get_stage_data(session_id: str):
    try:
        stage = harness.state_manager.get_current_stage(session_id)
        if not stage:
            raise HTTPException(status_code=404, detail="Stage not found")
        return stage
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{session_id}")
async def get_progress_data(session_id: str):
    try:
        profile = harness.state_manager.get_profile(session_id)
        accuracy = harness.state_manager.calculate_accuracy(session_id)
        
        progress_data = {
            "accuracy": accuracy,
            "knowledge_gaps": profile.get("knowledge_gaps", []) if profile else [],
            "learning_rhythm": profile.get("learning_rhythm", "medium") if profile else "medium",
            "confidence_index": profile.get("confidence_index", 0.5) if profile else 0.5
        }
        return progress_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
