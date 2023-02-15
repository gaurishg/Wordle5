#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <bitset>
#include <span>
#include <utility>
#include <iostream>
#include <ciso646>
#include <chrono>
#include <unordered_set>
#include <ranges>
#include <omp.h>

#define SHOW_OUTPUT

#ifdef SHOW_OUTPUT
#define OUTPUT(x) x;
#else
#define OUTPUT(x) ;
#endif

using std::array;
using std::string;
using std::string_view;
using std::vector;
using std::unordered_map;
using Bitset = std::bitset<32>;
using MapType = std::unordered_map<Bitset, std::vector<std::string>>;
using ResultType = vector<array<string_view, 5>>;
using GraphType = std::unordered_map<Bitset, vector<Bitset>>;

constexpr std::string_view input_filename {"words_alpha.txt"};
constexpr std::string_view output_filename {"output_omp.txt"};

inline Bitset word_to_bitmap(std::string_view word)
{
    Bitset b{};
    for (const int c: word)
        b.set(c - 'a');
    return b;
}

std::tuple<vector<Bitset>, vector<string_view>, MapType> process_input_file(std::string_view filename = input_filename)
{
    MapType m;
    vector<string_view> vec;
    vector<Bitset> v_bitset;
    std::unordered_set<Bitset> st;

    st.reserve(100'000);
    v_bitset.reserve(100'000);
    vec.reserve(100'000);
    m.reserve(100'000);

    std::ifstream in{std::string{filename}};
    std::string word;
    while (in >> word)
    {
        const auto btmp = word_to_bitmap(word);
        if (word.size() == 5 and btmp.count() == 5)
        {
            st.insert(btmp);
            vec.emplace_back(word);
            m[btmp].emplace_back(std::move(word));
        }
    }

    for (const auto b: st)
        v_bitset.push_back(b);
    return {std::move(v_bitset), std::move(vec), std::move(m)};
}

void get_result(const MapType& m, const std::array<Bitset, 5>& arr, ResultType& result)
{
    for (string_view w1: m.at(arr[0]))
        for (string_view w2: m.at(arr[1]))
            for (string_view w3: m.at(arr[2]))
                for (string_view w4: m.at(arr[3]))
                    for (string_view w5: m.at(arr[4]))
                        result.push_back({w1, w2, w3, w4, w5});
}

GraphType make_graph(std::span<const Bitset> vec)
{
    GraphType graph;
    graph.reserve(vec.size());
    for (const auto u: vec)
    {
        for (const auto v: vec)
        {
            if ((u & v).none())
            {
                graph[u].push_back(v);
                graph[v].push_back(u);
            }
        }
    }
    return graph;
}

void dfs(const MapType& m, const GraphType& graph, ResultType& result, const Bitset start, const int left, vector<Bitset>& v, std::unordered_set<Bitset>& done, const Bitset b, std::unordered_set<Bitset>& visited)
{
    if (left == 1)
    {
        OUTPUT(std::cout << "Found something" << std::endl;);
        get_result(m, {v[0], v[1], v[2], v[3], v[4]}, result);
        OUTPUT(std::cout << "Total solutions " << result.size() << std::endl;);
        return;
    }

    if (visited.contains(start))
        return;
    visited.insert(start);

    for (const auto& child: graph.at(start))
    {
        if (not done.contains(child) and not visited.contains(child) and (child & b).none())
        {
            v.push_back(child);
            dfs(m, graph, result, child, left - 1, v, done, child | b, visited);
            v.pop_back();
        }
    }
    visited.erase(start);
}

void find_words(const MapType& m, const GraphType& graph, ResultType& result)
{
    std::unordered_set<Bitset> done;
    std::vector<Bitset> acc;
    std::unordered_set<Bitset> visited;
    for (const auto& [b, vec_b]: graph)
    {
        acc.push_back(b);
        dfs(m, graph, result, b, 5, acc, done, b, visited);
        acc.pop_back();
        done.insert(b);
    }
}

void simply_nested_search(const MapType& m, std::span<const Bitset> v, ResultType& result)
{
    const int n = v.size();

    #pragma omp parallel for schedule(dynamic)
    for (int i1=0; i1<n; ++i1)
    {
        const auto b1 = v[i1];
        for (auto i2 = i1+1; i2 < n; ++i2)
        {
            const auto b2 = v[i2];
            if ((b1 | b2).count() != 10)
                continue;
            for (auto i3 = i2 + 1; i3 < n; ++i3)
            {
                const auto b3 = v[i3];
                if ((b1 | b2 | b3).count() != 15)
                    continue;
                for (auto i4 = i3 + 1; i4 < n; ++i4)
                {
                    const auto b4 = v[i4];
                    if ((b1 | b2 | b3 | b4).count() != 20)
                        continue;
                    for (auto i5 = i4 + 1; i5 < n; ++i5)
                    {
                        const auto b5 = v[i5];
                        if ((b1 | b2 | b3 | b4 | b5).count() != 25)
                            continue;
                            
                        #pragma omp critical
                            get_result(m, {b1, b2, b3, b4, b5}, result);
                        
                    }
                }
            }
        }
    }
}

int main()
{
    std::cout << "Programm running on " << omp_get_max_threads() << " threads" << std::endl;
    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto [v_bitset, v_strings, m] = process_input_file();
    const auto file_read_done_time = std::chrono::high_resolution_clock::now();
    std::cout << "File read in " << std::chrono::duration_cast<std::chrono::milliseconds>(file_read_done_time - start_time).count() << " milliseconds" << std::endl;
    std::cout << "Total words found " << v_strings.size() << std::endl;

    std::ofstream out{std::string(output_filename)};
    ResultType result;
    result.reserve(1'000);

    // const auto graph = make_graph(std::span(v_bitset.begin(), v_bitset.end()));

    std::cout << "Work started" << std::endl;
    simply_nested_search(m, std::span(v_bitset.cbegin(), v_bitset.cend()), result);
    // find_words(m, graph, result);
    std::cout << "Work ended" << std::endl;
    const auto dfs_done_time = std::chrono::high_resolution_clock::now();
    std::cout << "Work done in " << std::chrono::duration_cast<std::chrono::seconds>(dfs_done_time - file_read_done_time).count() << " seconds, " << result.size() << " solutions found" << std::endl;
    for (auto [w1, w2, w3, w4, w5]: result)
        out << w1 << ' ' << w2 << ' ' << w3 << ' ' << w4 << ' ' << w5 << '\n';
    const auto output_written_time = std::chrono::high_resolution_clock::now();
    std::cout << "File written in " << std::chrono::duration_cast<std::chrono::milliseconds>(output_written_time - dfs_done_time).count() << " milliseconds" << std::endl;
    std::cout << "Total time taken " << std::chrono::duration_cast<std::chrono::seconds>(output_written_time - start_time).count() << std::endl;
}