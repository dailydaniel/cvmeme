from pydantic import BaseModel


class CV(BaseModel):
    work_experience: int
    responsibility: int
    creativity: int
    communication_skills: int
    leadership_qualities: int
    stress_resistance: int
    job_difficulty_level: int
    average_length_of_employment_at_one_job: int
    career_growth_speed: int
    description_details_level: int
    description_complexity_level: int
    arrogance_level: int
    education_quality: int
    academic_level: int
    foreign_languages: int
    professional_certificates: int
