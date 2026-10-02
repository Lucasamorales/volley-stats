"""Data contracts passed between pipeline stages and stored per video."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Team(StrEnum):
    US = "us"
    THEM = "them"

    @property
    def other(self) -> "Team":
        return Team.THEM if self is Team.US else Team.US


class Side(StrEnum):
    """Half of the court relative to the camera."""

    NEAR = "near"
    FAR = "far"


class Action(StrEnum):
    SERVE = "serve"
    RECEIVE = "receive"
    DIG = "dig"
    SET = "set"
    ATTACK = "attack"
    BLOCK = "block"
    UNKNOWN = "unknown"


class VideoMeta(BaseModel):
    path: str
    fps: float
    width: int
    height: int
    n_frames: int


class BallPoint(BaseModel):
    frame: int
    x: float | None = None
    y: float | None = None
    conf: float = 0.0

    @property
    def visible(self) -> bool:
        return self.x is not None and self.y is not None


class PlayerObs(BaseModel):
    """One tracked player in one frame, in image pixels."""

    frame: int
    track_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    wrists: list[tuple[float, float]] = Field(default_factory=list)

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


class Contact(BaseModel):
    frame: int
    x: float
    y: float
    score: float = 0.0  # velocity change at the contact, px/frame
    track_id: int | None = None
    player: str | None = None  # roster name; None for opponents or unresolved
    team: Team | None = None
    court_xy: tuple[float, float] | None = None  # toucher's foot point, metres
    point: tuple[float, float] | None = None  # where a person clicked a toucher the detector missed, px
    action: Action = Action.UNKNOWN
    ambiguous: bool = False  # another player was almost as close to the ball
    reviewed: bool = False  # a person confirmed or edited this touch
    deleted: bool = False  # a person marked it as not a touch; kept so re-runs do not bring it back
    manual: bool = False  # added by a person, missed by detection


class Rally(BaseModel):
    index: int
    set_number: int = 1
    start_frame: int
    end_frame: int
    serving_team: Team | None = None
    serving_reviewed: bool = False
    winner: Team | None = None
    winner_reviewed: bool = False
    valid: bool | None = None  # False = a person marked it as not a rally (warm-up, dead ball)
    reviewed: bool = False  # a person finished checking this rally; re-runs leave it alone
    contacts: list[Contact] = Field(default_factory=list)

    @property
    def touches(self) -> list[Contact]:
        """Contacts that count: attributed to a team and not rejected."""
        return [c for c in self.contacts if c.team is not None and not c.deleted]


class StatEvent(BaseModel):
    set_number: int
    rally_index: int
    team: Team
    player: str | None
    kind: str  # an Action value, or a terminal outcome such as "kill" / "ace"


class SetConfig(BaseModel):
    set_number: int
    us_side: Side  # which half our team plays on in this set
    start_frame: int = 0
    end_frame: int | None = None  # None = to the end of the video


class MatchRecord(BaseModel):
    """Everything known about one video; serialised to data/processed/<video>/match.json."""

    video: VideoMeta
    court_points: dict[str, tuple[float, float]] = Field(default_factory=dict)  # landmark -> image px
    court_estimated: bool = False  # True until a person has clicked the landmarks
    homography: list[list[float]] | None = None  # image px -> court metres
    roster: dict[int, str] = Field(default_factory=dict)  # track_id -> player name
    sets: list[SetConfig] = Field(default_factory=list)
    rallies: list[Rally] = Field(default_factory=list)
