
function shuffle(array) {
  var currentIndex = array.length, temporaryValue, randomIndex ; 
  while (0 !== currentIndex) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }
  return array;
}
$(document).ready(function() {
	var filters_overwrite = '';
	var arr_categ = [ { 'categ' : 'Teen', 'name' : 'Teen' },{ 'categ' : 'Big Tit', 'name' : 'Big Tit' },{ 'categ' : 'Asian', 'name' : 'Asian' },{ 'categ' : 'Lesbian', 'name' : 'Lesbian' },{ 'categ' : 'Couples', 'name' : 'Couples' },{ 'categ' : 'Ebony', 'name' : 'Ebony' },{ 'categ' : 'BBW', 'name' : 'BBW' },{ 'categ' : 'Pornstars', 'name' : 'Pornstars' },{ 'categ' : 'Latina', 'name' : 'Latina' }, ];
	//shuffle(arr_categ);	
	for (var i = 0; i < arr_categ.length ; i++) {
		filters_overwrite+= '<a class="item type_category filter_purple" target="_new" href="#" data-category="'+ arr_categ[i]['categ'] +'"><span></span> '+ arr_categ[i]['name'] +'</a>';
	}
	$("#filters_overwrite").html(filters_overwrite);
});
