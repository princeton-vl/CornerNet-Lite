import pkg_resources

_package_name = __name__

def get_file_path(*paths):
    path = "/".join(paths)
    return pkg_resources.resource_filename(_package_name, path)
