import re
from guardrails.validators import Validator, register_validator

@register_validator(name="movie_related")
class MovieRelatedValidator(Validator):
    def validate(self, value, **kwargs):
        keywords = ["movie", "film", "genre", "actor", "director", "cinema", "screenplay", "cast", "poster", "trailer"]
        
        if any(keyword.lower() in value.lower() for keyword in keywords):
            return True
        else:
            return False

    def message(self, value, **kwargs):
        return "I'm only here to answer movie-related questions."
