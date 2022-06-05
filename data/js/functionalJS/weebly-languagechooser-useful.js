
function toggleLanguageChooser() {
	var lc = $('language-chooser');
	if(lc.hasClassName('language-open')) {
		hideLanguageChooser();
	} else {
		showLanguageChooser();
	}
}

function showLanguageChooser() {
	$('language-chooser').addClassName('language-open');
	document.stopObserving('click', handleLanguageChooserClick);
	document.observe('click', handleLanguageChooserClick);
}

function hideLanguageChooser() {
	$('language-chooser').removeClassName('language-open');
}

function handleLanguageChooserClick(event) {
	var el = Event.element(event);
	if(!(el.id == 'language-chooser' || el.up('#language-chooser'))) {
		hideLanguageChooser()
		document.stopObserving('click', handleLanguageChooserClick);
	}
}
