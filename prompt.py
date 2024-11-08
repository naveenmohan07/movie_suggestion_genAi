prompts = {
    "retrive_from_message": "Analyze the user's message based on the following context: {context}. Determine if they are asking for a **movie suggestion** or a **movie name** based on a story. Respond with **'movie_suggestion'** for recommendations, **'movie_name'** if they want a title from a story.",
    "movie_suggestion": "Given the following retrieved information: {context}, based on the user's request to suggest movies, provide up to two movies that match the specified genre. Include the movie name, overview, language, poster, and a link to the homepage. Avoid including markdown formatting",
    "movie_name": "Based on the user's message based on the following context: {context}, provide a movie name which has the similar story",
}
