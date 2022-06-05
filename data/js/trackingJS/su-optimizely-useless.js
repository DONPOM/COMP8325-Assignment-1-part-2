
  (function() {
  var appScriptEl, assets, branch, branches, defaultBranch, forceBranchParam, headEl, key, protocol, reason, requestedBranch, requestedBy, styleEl, suteamBranch, value, variant;

  defaultBranch = 'master';

  suteamBranch = 'suteam2';

  headEl = document.head || document.querySelector('head');

  appScriptEl = document.createElement('script');

  styleEl = document.createElement('link');

  protocol = window.location.protocol;

  if ((typeof optimizely !== "undefined" && optimizely !== null ? optimizely.activeExperiments.length : void 0) > 0) {
    variant = optimizely.variationNamesMap[optimizely.activeExperiments].toLowerCase();
  }

  branches = [];

  for (key in SUassets) {
    value = SUassets[key];
    branches.push(key);
  }

  forceBranchParam = 'SUbranch=';

  if (window.top.location.search.match(forceBranchParam)) {
    requestedBranch = window.top.location.search.split(forceBranchParam).pop();
    if (typeof cookie !== "undefined" && cookie !== null) {
      cookie.set('b', requestedBranch);
    }
    requestedBy = 'get param';
  } else if (typeof cookie !== "undefined" && cookie !== null ? cookie.get('b') : void 0) {
    requestedBranch = typeof cookie !== "undefined" && cookie !== null ? cookie.get('b') : void 0;
    requestedBy = 'cookie';
  } else if (typeof cookie !== "undefined" && cookie !== null ? cookie.get('suteam' === true) : void 0) {
    requestedBranch = suteamBranch;
    requestedBy = 'suteam';
  } else if (variant) {
    requestedBranch = variant;
    requestedBy = 'test';
  }

  if (requestedBranch) {
    console.debug("requesting branch because " + requestedBy, requestedBranch);
  }

  if (branches.length === 1) {
    branch = branches.shift();
    reason = 'only one';
  } else if (branches.indexOf(requestedBranch) !== -1) {
    branch = requestedBranch;
    reason = 'requested';
  } else {
    branch = defaultBranch;
    reason = 'default';
  }

  console.debug("using branch because " + reason, branch);

  assets = SUassets[branch][protocol];

  styleEl.setAttribute('rel', 'stylesheet');

  styleEl.setAttribute('href', assets.css);

  if (!(assets.app.match('origin.stumble.net|static'))) {
    appScriptEl.setAttribute('crossorigin', 'anonymous');
  }

  appScriptEl.setAttribute('src', assets.app);

  headEl.appendChild(styleEl);

  headEl.appendChild(appScriptEl);

  appScriptEl.addEventListener('error', function() {
    throw new Error("Unable to load " + assets.app + " from " + branch);
  }, false);

}).call(this);

