const movieField = document.getElementById("movie_field");
const userField = document.getElementById("user_id_field");
const numRecField = document.getElementById("num_recommend_field");
const topField = document.getElementById("top_field");
const genreField = document.getElementById("genre_field");
const yearField = document.getElementById("year_field");
const getTop = document.getElementById("top_rated");
const imdbRating = document.getElementById("imdb_rating");
const lensRating = document.getElementById("lens_rating");
const getInfo = document.getElementById("movie_info");
const recommend = document.getElementById("recommend");
const leftUpperResponse = document.getElementById("left_area_upper");
const leftLowerResponse = document.getElementById("left_area_lower");
const rightResponse = document.getElementById("right_area");

imdbRating.addEventListener("click",
    () => {sendRatingRequest("omdb")});
lensRating.addEventListener("click",
    () => {sendRatingRequest("movie_lens")});
getTop.addEventListener("click", sendTopRequest);
getInfo.addEventListener("click", sendInfoRequest);
recommend.addEventListener("click", sendRecommendations);

async function sendRecommendations() {
    const user_id = userField.value;
    const recommend_num = numRecField.value;

    const response = await fetch(`/recommend/${user_id}/${recommend_num}`, {
        method: "GET",
    });

    const titlesList = await response.json();

    if (titlesList.length === 0) {
        leftLowerResponse.innerHTML = "<li>No recommendations found for this user.</li>";
        return;
    }

    const listItems = titlesList.map(title => `
        <li class="rec-item">${title}</li>
    `);
    leftLowerResponse.innerHTML = `<ul>${listItems.join('')}</ul>`;
}

async function sendRatingRequest(database) {

    const title = movieField.value;

    const response = await fetch(`/${database}/${title}/rating`, {
        method: "GET",
    });

    const data = await response.json();
    if (data === "unavailable") {
        leftUpperResponse.innerHTML = "Try later";
        return;
    }
    leftUpperResponse.innerHTML = `Did you mean "${data.movie_title}"?<br>
                              Rating: ${data.rating}/10`;
}

async function sendTopRequest() {
    const n = topField.value;
    const genre = genreField.value;
    const year = yearField.value.toString();

    const response = await fetch(`/movie_lens/top/?n=${n}&genre=${genre}&release_year=${year}`, {
        method: "GET",
    });

    const moviesList = await response.json();
    if (moviesList === "unavailable") {
        rightResponse.innerHTML = "Try later";
        return;
    }

    const moviesDisplay = moviesList.map(movie =>
    {
        return `<li>
                    <strong>${movie.title}</strong><br>
                    Rating: ${movie.rating}/10 (${movie.reviews} reviews)<br>
                    Release year: ${movie.release_year}
                </li>`;
    });

    rightResponse.innerHTML = `Top-rated ${genre} movies with at least 100 reviews:
                                <ul>${moviesDisplay.join('')}</ul>`;
}

async function sendInfoRequest() {
    const movie_title = movieField.value;

    const response = await fetch(`/movies/${movie_title}`, {
        method: "GET",
    });

    const infoList = await response.json();
    if (infoList === "unavailable") {
        leftUpperResponse.innerHTML = "Try later";
        return;
    }
    const display = infoList.map(([label, value]) =>
    {
        return `<li>
                    <strong>${label}</strong>: ${value}
                </li>`;
    });

    leftUpperResponse.innerHTML = `<ul>${display.join('')}</ul>`;
}
