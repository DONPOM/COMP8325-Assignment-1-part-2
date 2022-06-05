
$j(document).ready(function() {
	var searchInputs = document.getElementsByName('s');
	for (var i = 0; i < searchInputs.length; ++i) {
		if ( (!searchInputs[i].placeholder) && !($j(searchInputs[i]).hasClass('no-mod-gcs')) ) {
			if (searchInputs[i].getAttribute('placeholder')) {
				searchInputs[i].placeholder = searchInputs[i].getAttribute('placeholder');
			} else {
				searchInputs[i].placeholder = searchInputs[i].value;
			}
		}
		if ( (supportsInputPlaceholder) && !($j(searchInputs[i]).hasClass('no-mod-gcs')) ) {
			searchInputs[i].value = '';
		}
		searchInputs[i].onclick = function(){};
		setInputPlaceholderFallback(searchInputs[i]);
		var form = searchInputs[i].form;
		if (!form.action.match(/search/))
			continue;
		searchInputs[i].setAttribute('autocomplete', 'off');
		searchInputs[i].setAttribute('spellcheck', 'false');
				form.action = form.action.replace('search', 'searchx');
	}
});
