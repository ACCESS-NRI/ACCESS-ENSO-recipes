const REPOSITORY = 'https://github.com/ACCESS-NRI/ACCESS-Evaluation-Recipes';
const form = document.querySelector('#filters');
const fields = {
  q: document.querySelector('#search'),
  subject: document.querySelector('#subject'),
  realm: document.querySelector('#realm'),
  tool: document.querySelector('#tool'),
  kind: document.querySelector('#kind'),
};
const results = document.querySelector('#results');
const resultCount = document.querySelector('#result-count');
const emptyState = document.querySelector('#empty-state');
let recipes = [];

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function optionsFor(field, values) {
  for (const value of [...new Set(values)].sort((a, b) => a.localeCompare(b))) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    field.append(option);
  }
}

function cardFor(recipe) {
  const card = node('article', 'recipe-card');
  const image = node('div', `card-image${recipe.preview ? '' : ' placeholder'}`);
  if (recipe.preview) {
    const preview = node('img');
    preview.src = recipe.preview;
    preview.alt = `Saved output from ${recipe.title}`;
    preview.loading = 'lazy';
    image.append(preview, node('span', 'image-note', 'Saved notebook output'));
  } else {
    image.append(node('span', 'placeholder-label', recipe.subjects[0] || 'Recipe'));
  }
  card.append(image);

  const body = node('div', 'card-body');
  const topline = node('div', 'card-topline');
  topline.append(node('span', '', recipe.collection), node('span', 'card-kind', recipe.status || recipe.kind));
  body.append(topline, node('h3', '', recipe.title), node('p', 'card-summary', recipe.summary));
  const tags = node('div', 'card-tags');
  for (const value of [...new Set([...recipe.subjects, ...recipe.realms, ...recipe.tools])].slice(0, 4)) {
    tags.append(node('span', '', value));
  }
  body.append(tags);
  const link = node('a', 'card-link');
  link.href = `${REPOSITORY}/blob/main/${recipe.path.split('/').map(encodeURIComponent).join('/')}`;
  link.setAttribute('aria-label', `Open ${recipe.title} in the repository`);
  link.append(document.createTextNode(recipe.kind === 'ESMValTool recipe' ? 'View YAML recipe' : 'Open notebook'), node('span', '', '↗'));
  body.append(link);
  card.append(body);
  return card;
}

function selected() {
  return Object.fromEntries(Object.entries(fields).map(([name, field]) => [name, field.value.trim()]));
}

function matching(recipe, filters) {
  if (filters.subject && !recipe.subjects.includes(filters.subject)) return false;
  if (filters.realm && !recipe.realms.includes(filters.realm)) return false;
  if (filters.tool && !recipe.tools.includes(filters.tool)) return false;
  if (filters.kind && recipe.kind !== filters.kind) return false;
  if (!filters.q) return true;
  const haystack = [recipe.title, recipe.summary, recipe.collection, ...recipe.subjects, ...recipe.realms, ...recipe.tools].join(' ').toLocaleLowerCase();
  return filters.q.toLocaleLowerCase().split(/\s+/).every(term => haystack.includes(term));
}

function render() {
  const filters = selected();
  const matches = recipes.filter(recipe => matching(recipe, filters));
  results.replaceChildren(...matches.map(cardFor));
  emptyState.hidden = matches.length !== 0;
  resultCount.textContent = `${matches.length} of ${recipes.length} recipes`;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) if (value) params.set(key, value);
  const search = params.toString();
  history.replaceState(null, '', `${location.pathname}${search ? `?${search}` : ''}${location.hash}`);
}

function reset() {
  form.reset();
  requestAnimationFrame(render);
}

async function init() {
  try {
    const response = await fetch('catalogue.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    recipes = await response.json();
    optionsFor(fields.subject, recipes.flatMap(recipe => recipe.subjects));
    optionsFor(fields.realm, recipes.flatMap(recipe => recipe.realms));
    optionsFor(fields.tool, recipes.flatMap(recipe => recipe.tools));
    optionsFor(fields.kind, recipes.map(recipe => recipe.kind));
    const params = new URLSearchParams(location.search);
    for (const [key, field] of Object.entries(fields)) {
      const value = params.get(key);
      if (value) field.value = value;
    }
    document.querySelector('#stat-recipes').textContent = recipes.length;
    document.querySelector('#stat-subjects').textContent = new Set(recipes.flatMap(recipe => recipe.subjects)).size;
    document.querySelector('#stat-previews').textContent = recipes.filter(recipe => recipe.preview).length;
    form.addEventListener('input', render);
    form.addEventListener('change', render);
    form.addEventListener('reset', () => requestAnimationFrame(render));
    document.querySelector('#empty-reset').addEventListener('click', reset);
    render();
  } catch (error) {
    resultCount.textContent = 'Catalogue unavailable';
    emptyState.hidden = false;
    emptyState.querySelector('h3').textContent = 'Could not load recipes';
    emptyState.querySelector('p').textContent = 'Please try again later or browse the repository.';
    console.error(error);
  }
}

init();
