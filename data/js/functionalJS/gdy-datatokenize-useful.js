
$(document).ready(function () {
$('[data-tokenize]').each(function(){
var $this = $(this),
html = $this.html(),
val = $this.data('tokenize'),
tokenizedHtml = html.replace(/\{0\}/gi, val);
$this
.html(tokenizedHtml)
});
});
