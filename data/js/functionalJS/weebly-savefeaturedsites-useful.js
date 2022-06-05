
		function saveFeaturedSiteRequest()
		{
			if( $('featured-site-id').value.length == 0 )
			{
				showErrorMessage( 'Please select a site', 'featured-site-id', 'right' );
			}
			else if( $('featured-site-description').value.length == 0 )
			{
				showErrorMessage( 'Please describe your site', 'featured-site-description', 'right' );
			}
			else
			{
				new Ajax.Request(ajax, {
					parameters: {
						pos: 'featuredSiteRequest',
						site_id: $('featured-site-id').value,
						title: $('featured-site-title').value,
						description: $('featured-site-description').value
					},
					onComplete: function(){
						showErrorMessage('Your request has been received and is pending review.', 'featured-site-button', 'right');
					}
				});
			}
		}

		jQuery(document).ready(function(){
			if (jQuery.fn.wCarousel) {
				jQuery('#promo-carousel').wCarousel();
			}
		});

		Weebly.ready(function() {

			Weebly.Sites = {};
						Weebly.Sites['833818312774124822'] = {"site_id":"833818312774124822","user_id":"43548859","site_title":"My Site","user_identifier":"998454723627745399","custom_domain":"","site_version":"0","category":"","theme":"758493340479023180"};
			Weebly.Sites['833818312774124822']['level'] = '0';
			Weebly.Sites['833818312774124822']['hasTrial'] = false;
			Weebly.Sites['833818312774124822']['hasSSL'] = '';
						
			Weebly.RequestToken = 'be6225da6e5bcf18f6ca110a218ef5eb';

		}, true); // highPriority=true

	