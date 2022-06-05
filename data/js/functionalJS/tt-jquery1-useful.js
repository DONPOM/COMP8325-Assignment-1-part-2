
    jQuery(document).ready(function() {
        jQuery('#search-form div.submit-search').click(function(e) {
            e.preventDefault();
            e.stopPropagation();
            jQuery('#search-form').submit();
        });
    });
