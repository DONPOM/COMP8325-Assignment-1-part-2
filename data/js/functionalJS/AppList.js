define("site/appstore/AppList", [
	'jquery',
	'underscore',
	'backbone',
	'instance/site/site',
	'site/appstore/App'
], function($, _, Backbone, site, App){
	var AppList = Backbone.Collection.extend({
		model: App,

		initialize: function(){
			var self = this;
			this.bind("change:installed", function(app){
				app.set("show", app.is(self.currentFilter));
			});
			site.fetch({cache: false}).done(function(site){
				self.social = site.social;
			});
		},

		setFeatured: function(){
			if (!this.models.length) return false;

			var apps = _.map(this.where({featured:true}), function(app){
				return app.set('featured', false);
			}, this);

			if(apps.length === 0) return false;

			apps[Math.floor(Math.random() * apps.length)].set('featured',true);
		},

		sortByFeatured: function(){
			return _(this.sortBy(function(app){
				return !app.get('featured') + app.get('name');
			}));
		},

		fetchAll: function(){
			if(this.fetched) return;

			this.fetch({data: {'showUninstalledApps': true}});
			this.fetched = true;
		},

		appFilter: function(name){
			this.currentFilter = name;
			_.each(this.models, function(app){
				app.set('show', app.is(name));
			});
		},

		isAppInstalled: function (appHandle) {
			var findResult = this.find(function (app) {
				return app.get('handle') == appHandle;
			});

			return !!findResult;
		}
	});

	return AppList;
});
