
$(function() {
	if (readCookie('adminuser')) {
		$('body').append('<span id="edit-page-btn" style="display: block; padding: 10px; z-index: 9999; background: #000; color: #fff; font-size: 12px; position: fixed; right: 0; top: 0; cursor: default">Edit Page</span>');
		$('#edit-page-btn').click(function() {
			url = document.location.protocol + '//' + document.location.hostname
			if (document.location.port) url += ':' + document.location.port;
			url += '/admin/content/wiki/g2g';
			url += document.location.pathname;
			window.location = url;
		});
	}
});
