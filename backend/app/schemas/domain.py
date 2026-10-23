"""Domain schemas shared across agents, services and the API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
AgentName = Literal["orchestrator", "planner", "research", "browser", "workspace", "verification", "recovery", "messenger"]
PricingStatus = Literal["paid", "free", "freemium", "contact_sales", "not_found"]
OutputKind = Literal["xlsx", "pptx"]

MAX_COMPETITORS = 10


class Intent(BaseModel):
    """Structured result of the Intent Engine."""

    feasible: bool = True
    rejection_reason: str | None = None
    workflow_type: str = "research_report"  # research_report | test_paper
    goal: str = ""
    industry: str = ""
    competitor_count: int = 5
    outputs: list[OutputKind] = Field(default_factory=lambda: ["xlsx", "pptx"])
    # test_paper fields
    subject: str = ""
    grade: int | None = None
    topics: list[str] = Field(default_factory=list)
    num_questions: int = 20
    difficulty: str = "mixed"
    duration_minutes: int | None = None
    post_to_telegram: bool = False
    num_papers: int = 1
    total_marks: int | None = None
    inputs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "LOW"

    @model_validator(mode="after")
    def _check(self) -> "Intent":
        if self.feasible and self.workflow_type == "test_paper":
            if not self.subject.strip() or self.grade is None:
                raise ValueError("subject and grade are required for a test paper")
        elif self.feasible and self.workflow_type == "research_report":
            if not self.industry.strip():
                raise ValueError("industry is required for a feasible intent")
            if not 1 <= self.competitor_count <= MAX_COMPETITORS:
                raise ValueError(f"competitor_count must be between 1 and {MAX_COMPETITORS}")
            if not self.outputs:
                raise ValueError("at least one output (xlsx/pptx) is required")
        elif not self.feasible and not self.rejection_reason:
            raise ValueError("an infeasible intent must explain why")
        return self


class WorkflowParams(BaseModel):
    """Validated parameters of the research_report workflow."""

    industry: str = Field(min_length=2, max_length=120)
    competitor_count: int = Field(5, ge=1, le=MAX_COMPETITORS)
    outputs: list[OutputKind] = Field(default_factory=lambda: ["xlsx", "pptx"], min_length=1)
    export_name: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,48}$")
    has_inputs: bool = False

    @field_validator("industry")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 2:
            raise ValueError("industry is too short")
        return v

    @field_validator("outputs")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class ExamParams(BaseModel):
    """Validated parameters of the test_paper workflow."""

    subject: str = Field(min_length=2, max_length=60)
    grade: int = Field(ge=1, le=12)
    topics: list[str] = Field(default_factory=list, max_length=12)
    num_questions: int = Field(20, ge=5, le=50)
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "mixed"
    duration_minutes: int = Field(60, ge=10, le=240)
    post_to_telegram: bool = False
    dry_run: bool = False  # simulated runs never contact Telegram
    num_papers: int = Field(1, ge=1, le=10)  # how many different tests to create
    total_marks: int | None = Field(None, ge=5, le=200)  # exact marks per paper; None = the default blueprint
    folder: str | None = Field(None, max_length=80)  # organised output folder name (derived when omitted)

    @field_validator("subject")
    @classmethod
    def _norm(cls, v: str) -> str:
        return " ".join(v.split())


class Question(BaseModel):
    id: str
    section: Literal["A", "B", "C"]
    type: Literal["mcq", "short", "long"]
    text: str = Field(min_length=8, max_length=900)
    options: list[str] = Field(default_factory=list, max_length=12)  # exactly-4 is enforced (with feedback) by normalise_paper
    answer: str = Field(min_length=1, max_length=300)  # MCQ: option letter A-D
    final_answer: str = ""  # machine-comparable final value (e.g. "x = 3 or x = -2")
    solution: str = Field(min_length=3, max_length=1500)
    marks: int = Field(ge=1, le=20)
    topic: str = ""


class ExamPaper(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    instructions: list[str] = Field(default_factory=list, max_length=10)
    questions: list[Question] = Field(min_length=1, max_length=60)


class SolvedAnswer(BaseModel):
    id: str
    answer: str  # MCQ: option letter; otherwise the final value only


class SolvedSet(BaseModel):
    answers: list[SolvedAnswer]


class TaskSpec(BaseModel):
    key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_\-]{0,30}$")
    title: str = Field(min_length=3, max_length=200)
    description: str = ""
    agent: AgentName
    action: str
    params: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    expected_output: str = ""
    verification_criteria: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "LOW"


# --- research data model -------------------------------------------------------------------

class Candidate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    url: str
    rationale: str = ""


class CandidateList(BaseModel):
    competitors: list[Candidate] = Field(min_length=1, max_length=20)


class PageRecord(BaseModel):
    url: str
    final_url: str = ""
    title: str = ""
    text: str = ""
    retrieved_at: datetime
    via: Literal["http", "browser", "demo"] = "http"
    role: Literal["homepage", "pricing", "corroboration"] = "homepage"
    status_code: int | None = None
    links: list[dict] = Field(default_factory=list)  # [{"text":..., "url":...}]
    screenshot_artifact_id: str | None = None
    simulated: bool = False


class FeatureClaim(BaseModel):
    text: str = Field(min_length=3, max_length=300)
    evidence: str = Field(min_length=3, max_length=600)
    source_url: str = ""


class PlanClaim(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    price_text: str = Field(min_length=1, max_length=80)  # exact text, e.g. "$29 per user / month"
    price_usd: float | None = Field(None, ge=0)
    billing: str = ""
    evidence: str = Field(min_length=1, max_length=600)
    source_url: str = ""


class ExtractedFacts(BaseModel):
    """What the extraction model must return (validated deterministically afterwards)."""

    product_name: str = ""
    summary: str = ""
    summary_evidence: str = ""
    features: list[FeatureClaim] = Field(default_factory=list, max_length=12)
    pricing_status: PricingStatus = "not_found"
    plans: list[PlanClaim] = Field(default_factory=list, max_length=8)
    pricing_note: str = ""


class Feature(BaseModel):
    text: str
    evidence_id: str


class Plan(BaseModel):
    name: str
    price_text: str
    price_usd: float | None
    billing: str = ""
    evidence_id: str


class SourceRef(BaseModel):
    url: str
    title: str = ""
    retrieved_at: datetime
    role: str = "homepage"


class CompetitorRecord(BaseModel):
    """Normalised, evidence-linked competitor row used by Excel/PowerPoint generation."""

    name: str
    product: str
    website: str
    summary: str = ""
    features: list[Feature] = Field(default_factory=list)
    pricing_status: PricingStatus = "not_found"
    plans: list[Plan] = Field(default_factory=list)
    pricing_note: str = ""
    sources: list[SourceRef] = Field(default_factory=list)
    verification: Literal["verified", "partial", "unverified"] = "unverified"
    simulated: bool = False

    @property
    def starting_price_usd(self) -> float | None:
        prices = [p.price_usd for p in self.plans if p.price_usd is not None and p.price_usd > 0]
        return min(prices) if prices else None


# --- API payloads --------------------------------------------------------------------------

class ObjectiveCreate(BaseModel):
    objective: str = Field(min_length=3, max_length=4000)
    mode: Literal["guided", "autonomous"] = "guided"
    simulate: bool = False
    upload_ids: list[str] = Field(default_factory=list, max_length=5)
    export_name: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,48}$")


class PlanUpdate(BaseModel):
    tasks: list[TaskSpec]


class PlanApprove(BaseModel):
    plan_hash: str | None = Field(None, max_length=64)  # the hash of the plan version the approver reviewed


class ExternalActionResolve(BaseModel):
    outcome: Literal["performed", "not_performed"]
    note: str | None = Field(None, max_length=500)


class ApprovalResolve(BaseModel):
    decision: Literal["approved", "denied"]
    note: str | None = Field(None, max_length=500)


class WorkflowCreate(BaseModel):
    run_id: str
    name: str = Field(min_length=2, max_length=120)
    description: str = Field("", max_length=500)


class WorkflowRunRequest(BaseModel):
    params: dict = Field(default_factory=dict)
    mode: Literal["guided", "autonomous"] = "guided"
    simulate: bool | None = None


class SettingsUpdate(BaseModel):
    max_retries: int | None = Field(None, ge=0, le=5)
    max_parallel_tasks: int | None = Field(None, ge=1, le=8)
    http_timeout_seconds: float | None = Field(None, ge=2, le=120)
    task_timeout_seconds: float | None = Field(None, ge=10, le=1800)
    browser_enabled: bool | None = None
    browser_headless: bool | None = None
    auto_approve_medium: bool | None = None
    output_dir: str | None = Field(None, min_length=1, max_length=300)
