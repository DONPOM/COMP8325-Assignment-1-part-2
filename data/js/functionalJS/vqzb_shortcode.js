jQuery.noConflict();
jQuery(document).ready(function(jQuery){

    var overlay = new ItpOverlay("vqzb_wrapper");
    //get remained questions qty
    var remained_questions_qty = jQuery("#vqzb_table tr").length - 1;
    
    //show just fist question
    jQuery("#vqzb_table tr").hide();
    jQuery("#vqzb_table tr").first().show();
    
    //show the remained questions qty
    vqzb_set_quest_num(remained_questions_qty);
    
    jQuery("[name^=answers]").change(function(){
	// if it was the last question than submit form
	if (remained_questions_qty === 0){
	    vqzb_submit_form();
	}else{
	    remained_questions_qty --;
	    //show the remained questions qty
	    vqzb_set_quest_num(remained_questions_qty);
	    
	    //hide this question & show the next one
	    var cur_obj = jQuery(this).closest("tr");
	    cur_obj.hide();
	    cur_obj.next("tr").fadeIn();
	}
    });
    
function vqzb_submit_form()
{
    overlay.show();
    var ajax_options = {
        //dataType:  'json',
        data: {
            quiz_url : document.URL,
            action : 'vqzb_ajax'
        },
        success: function(data){
            overlay.hide();
            jQuery("#vqzb_content").html(data);
        },
        error: function(){
            overlay.hide();
            jQuery("#vqzb_content").html("<p>Results processing error!<p>");
        }
    };
    jQuery("#vqzb_shortcode_form").ajaxSubmit(ajax_options);
    return false;
}    
});



// Show the remained questions qty or "final question" in #vqzb_question_remain block
function vqzb_set_quest_num(num){
    if (num === 0){
	jQuery("#vqzb_question_remain").text('final question');
    }else{
	jQuery("#vqzb_question_remain").text(num + ' questions remain');
    }
}