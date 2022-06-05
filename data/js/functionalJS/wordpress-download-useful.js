
(function($){
$(document).ready(function() {
	$('#footer a').click(function() {
		if (this.href.indexOf('wordpress.org') == -1 && this.href.indexOf('http') == 0) {
			recordOutboundLink(this, 'Outbound Links', this.href);
			return false;
		}
	});
	$('#download a, a.download-button').click(function() {
		recordOutboundLink(this, 'Download Links', $(this).hasClass('download-button') ? 'button' : 'nav' );
		return false;
	});
});
})(jQuery);
