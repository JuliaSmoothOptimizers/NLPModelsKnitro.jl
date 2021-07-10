using Pkg
Pkg.activate(joinpath("temp_env"))
Pkg.add(["Git", "GitHub", "JSON"])
Pkg.instantiate()

using Git, GitHub, JSON

TEST_RESULTS_FILE = "test_results.txt"
TEST_RESULTS_JSON = "test_results.json"

# Need to add GITHUB_AUTH to your .bashrc
myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])

function create_gist(authentication)
  file_content = ""
  file = open(TEST_RESULTS_FILE, "r")
  for line in readlines(file)
    file_content *= line * '\n'
  end
  close(file)

  file_dict = Dict(TEST_RESULTS_FILE => Dict("content" => file_content))
  gist = Dict{String, Any}("description" => "Test results", "public" => true, "files" => file_dict)

  posted_gist = GitHub.create_gist(params = gist, auth = authentication)

  return posted_gist
end

function post_gist_url_to_pr(comment::String; kwargs...)
  api = GitHub.DEFAULT_API
  repo = get_repo(api, ENV["org"], ENV["repo"]; kwargs...)
  pull_request = get_pull_request(api, ENV["org"], repo, parse(Int, ENV["pullrequest"]); kwargs...)
  GitHub.create_comment(api, repo, pull_request, comment; kwargs...)
end

function get_repo(api::GitHub.GitHubWebAPI, org::String, repo_name::String; kwargs...)
  my_params = Dict(:visibility => "all")
  # return GitHub.repo(api, repo; params = my_params, kwargs...)
  return Repo(GitHub.gh_get_json(api, "/repos/$(org)/$(repo_name)"; params = my_params, kwargs...))
end

function get_pull_request(
  api::GitHub.GitHubWebAPI,
  org::String,
  repo::Repo,
  pullrequest_id;
  kwargs...,
)
  my_params = Dict(:sort => "popularity", :direction => "desc")
  pull_request = PullRequest(
    GitHub.gh_get_json(
      api,
      "/repos/$(org)/$(repo.name)/pulls/$(pullrequest_id)";
      params = my_params,
      kwargs...,
    ),
  )

  return pull_request
end

post_gist_url_to_pr("Here are the test results: $(create_gist(myauth).html_url)"; auth = myauth)
