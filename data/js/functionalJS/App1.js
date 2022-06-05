define("site/appstore/App", [
	'jquery',
	'underscore',
	'backbone'
], function($, _, Backbone){
	var App = Backbone.Model.extend({

		is: function(filter){
			if(filter === "webs") return this.isWebs();
			if(filter === "installed") return this.isInstalled();
			return true;
		},

		isWebs: function(){
			return this.get('developer_name').toLowerCase() === 'webs';
		},

		isInstalled: function(){
			return this.get("installed");
		},

		save: function(){
			throw new Error("Cannot save app information. Please use install or uninstall instead.");
		},

		install: function(opts){
			var self = this;
			return $.ajax({
				type: 'POST',
				url: this.collection.url,
				data: JSON.stringify({ id: this.get("id") }),
				processData: false,
				dataType: 'json',
				contentType: "application/json",

				success: function(data, status, xhr){
					self.set("installed", true);
					if(opts.success) opts.success(self, data);
				},
				error: function(data, status, xhr){
					self.set("installed", false);
					if(opts.error) opts.error(self, data);
				}
			});
		},

		uninstall: function(options){
			// NOTE: Code modified from Model.destroy in Backbone
			options = options ? _.clone(options) : {};

			var model = this;
			var pages = model.getPages();
			var success = options.success;

			if (model.collection.pageList.length < 2 && model.getPages().length) {
				model.trigger('cancelUninstall');
				throw new Error("cannot delete last page");
			}

			var triggerUninstall = function(){
				model.set("installed", false);
			};

			options.success = function(resp){
				if(options.wait) triggerUninstall();

				_.each(pages, function(page){
					if (page.collection !== undefined) {
						page.collection.removePage(page, {fromApp: true});
					}
				});

				if(success){
					success(model, resp);
				} else {
					model.trigger('sync', model, resp, options);
				}
			};

			options.error = function(){
				model.set("installed", true);
			};

			var xhr = (this.sync || Backbone.sync).call(this, 'delete', this, options);
			if(!options.wait) triggerUninstall();
			return xhr;
		},

		/**
		 * Warning: If we're not part of a collection, this will return an empty array.
		 */
		getPages: function(){
			return (this.collection ? this.collection.pageList.getAppPages(this.get('id')) : []);
		}
	});

	return App;
});
