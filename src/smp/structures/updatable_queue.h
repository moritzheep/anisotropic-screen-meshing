#pragma once

#include <unordered_map>
#include <vector>
#include <pmp/SurfaceMesh.h>

namespace std
{
    template<>
    struct hash<pmp::Vertex> : hash<pmp::IndexType>
    {
        inline size_t operator()(pmp::Vertex v) const
        {
            return hash<pmp::IndexType>::operator()(v.idx());
        }
    };

    template<>
    struct hash<pmp::Halfedge> : hash<pmp::IndexType>
    {
        inline size_t operator()(pmp::Halfedge h) const
        {
            return hash<pmp::IndexType>::operator()(h.idx());
        }
    };

    template<>
    struct hash<pmp::Edge> : hash<pmp::IndexType>
    {
        inline size_t operator()(pmp::Edge e) const
        {
            return hash<pmp::IndexType>::operator()(e.idx());
        }
    };

    template<>
    struct hash<pmp::Face> : hash<pmp::IndexType>
    {
        inline size_t operator()(pmp::Face f) const
        {
            return hash<pmp::IndexType>::operator()(f.idx());
        }
    };
}

namespace pmp
{
    template <typename Key, typename Value, typename Container = std::vector<std::pair<Key, Value>>, typename Compare = std::less<Value>>
    class updatable_queue
    {
        // We use an implementation of a binary heap but keep track of a mapping Vertex/Edge/Face <-> Heap Element

    private:
        using Node = typename Container::size_type;

        SurfaceMesh& mesh_;

        // The data structure containing the Key-Value Pairs
        Container container_;

        // The map keeping track of the position of each Key in the Container 
        std::unordered_map<Key, Node> map_;
        Compare compare_;

        // Get the Node representing the Key
        inline Node node(Key k) const
        {
            return map_.at(k);
        }

        // Get the Key that is represented by Node n
        inline Key key(Node n) const
        {
            return container_[n].first;
        }

        inline Value value(Node n) const
        {
            return container_[n].second;
        }

        inline Value& value(Node n)
        {
            return container_[n].second;
        }

        inline Value value(Key k) const
        {
            return value(node(k));
        }

        inline Value& value(Key k)
        {
            return value(node(k));
        }

        inline void swap(Node a, Node b)
        {
            // Swap both the elements in the heap and the map from Vertex/Edge/Face -> Heap Element
            std::swap(map_[key(a)], map_[key(b)]);
            std::swap(container_[a], container_[b]);
        }

        inline Node parent(Node n) const
        {
            return (n - 1) / 2;
        }

        inline Node left(Node n) const
        {
            return 2 * n + 1;
        }

        inline Node right(Node n) const
        {
            return 2 * n + 2;
        }

        inline bool is_valid(Node n) const
        {
            return n < container_.size();
        }

        inline bool has_left(Node n) const
        {
            return is_valid(left(n));
        }

        inline bool has_right(Node n) const
        {
            return is_valid(left(n));
        }

        // Returns whether this arangement of parent and child is valid for the type of heap
        inline bool is_heap(Node parent, Node child) const
        {
            // For a min heap, this is !(value(parent) < value(child)) <=> value(parent) >= value(child)
            return !compare_(value(parent), value(child));
        }

        void remove_from_heap(Node n)
        {
            // Swap element to be deleted and last element
            swap(n, container_.size() - 1);
            container_.pop_back();

            upheap(n);
            downheap(n);
        }

        void upheap(Node n)
        {
            while(n > 0 && !is_heap(parent(n), n))
            {
                swap(n, parent(n));
                n = parent(n);
            }
        }

        void downheap(Node n)
        {
            Node i = n;
            int ext, j;

            while(true)
            {
                ext = i;

                j = left(i);
                if (is_valid(j) && !is_heap(ext, j))
                {
                    ext = j;
                }

                j = right(i);
                if (is_valid(j) && !is_heap(ext, j))
                {
                    ext = j;
                }

                if (ext == i)
                {
                    break;
                }

                swap(ext, i);
                i = ext;
            }
        }

        // Position of the first Node in the level
        static constexpr Node begin(int level)
        {
            return std::pow(2, level) - 1;
        }

        // Position of the first node not in the level
        static constexpr Node end(int level)
        {
            return begin(level + 1);
        }

    public:
        updatable_queue(SurfaceMesh mesh) : mesh_(mesh)
        {

        }

        // Read access like a Vertex/Edge/Face Property
        const Value& operator[] (Key k) const
        {
            return value(node(k));
        }

        bool empty() const
        {
            return container_.empty();
        }

        typename Container::size_type size() const
        {
            return container_.size();
        }

        void reserve(typename Container::size_type new_cap)
        {
            container_.reserve(new_cap);
            map_.reserve(new_cap);
        }

        bool contains(Key k) const
        {
            return (map_.count(k) == 1);
        }

        Node levels() const
        {
            return std::ceil(std::log2(size() - 1));
        }

        // Unlike a Vertex/Edge/Face Property, writing triggers an update
        void update(Key k, const Value& v)
        {
            assert (map_.count(k) > 0 && "Key must already be in the heap");
            Node n = node(k);
            value(n) = v;
            upheap(n);
            downheap(n);
        } 

        void push(Key k, const Value& v)
        {
            assert(map_.count(k) == 0 && "Key is already in use.");
            Node n = container_.size();
            container_.push_back(std::make_pair(k, v));
            map_[k] = n;
            upheap(n);
        }

        // Must be followed by a call to heapify!
        void lazy_push(Key k, const Value& v)
        {
            assert(map_.count(k) == 0 && "Key is already in use.");
            Node n = container_.size();
            container_.push_back(std::make_pair(k, v));
            map_[k] = n;
        }

        void update_or_push(Key k, const Value& v)
        {
            if (map_.count(k) == 0) // Does not yet contain element
            {
                push(k, v);
            }
            else
            {
                update(k, v);
            }
        }

        void remove(Key k)
        {
            if (map_.count(k) > 0)
            {
                Node n = node(k);

                remove_from_heap(n);

                map_.erase(k);
            }
        }

        void pop()
        {
            if (size() > 1)
            {
                Key k = key(0);
                remove_from_heap(0);
                map_.erase(k);
            }
            else
            {
                container_.clear();
                map_.clear();
            }
        }

        const std::pair<Key, Value> top() const
        {
            return container_.front();
        }

        // Heapify is O(n) compared to O(n*log(n)) for inserting one by one
        void heapify()
        {
            if (size() > 1)
            {
                // for (int i = size() / 2 - 1; i != 0; --i)
                // {
                //     downheap(i);
                // }

                // Parallelize the downheap within each level
                for (int l = levels() - 2; l >= 0; --l)
                {
                    #pragma omp parallal for
                    for (auto i = begin(l); i != end(l); ++i)
                    {
                        downheap(i);
                    }
                }
            }
        }
    };
}