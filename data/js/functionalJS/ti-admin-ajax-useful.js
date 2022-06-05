
    

var td_blocks = []; //here we store all the items for the current page

//td_block class - each ajax block uses a object of this class for requests
function td_block() {
    this.id = '';
    this.block_type = 1; //block type id (1-234 etc)
    this.atts = '';
    this.td_cur_cat = '';
    this.td_column_number = '';
    this.td_current_page = 1; //
    this.post_count = 0; //from wp
    this.found_posts = 0; //from wp
    this.max_num_pages = 0; //from wp
    this.is_ajax_running = false;
    this.header_color = '';
}

    
var td_ad_background_click_link="";
var td_ad_background_click_target="";
var td_ajax_url="http://technorati.com/wp-admin/admin-ajax.php";
var td_get_template_directory_uri="http://technorati.com/wp-content/themes/Newspaper";
var tds_snap_menu="snap";
var tds_header_style="";
var tds_mobile_swipe="";
var td_search_url="http://technorati.com/search/";
var td_please_wait="Please wait...";
var td_email_user_pass_incorrect="User or password incorrect!";
var td_email_user_incorrect="Email or username incorrect!";
var td_email_incorrect="Email incorrect!";
var tds_more_articles_on_post_enable="show";
var tds_more_articles_on_post_time_to_wait="1";
var tds_more_articles_on_post_pages_distance_from_top="600";
