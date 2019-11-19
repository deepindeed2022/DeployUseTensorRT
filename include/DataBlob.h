#ifndef DEPLOY_INCLUDE_DataBlob_H_
#define DEPLOY_INCLUDE_DataBlob_H_
#include <utility>
#include <memory>
#include <iostream>
class Size {
public:
	Size(size_t h, size_t w):
		m_height(h), m_width(w)
	{}
	Size():
		m_height(0), m_width(0)
	{}

	size_t heights() const { return m_height; }
	size_t &heights() { return m_height; }
	size_t widths() const { return m_width; }
	size_t &widths() { return m_width; }

	bool operator == (const Size &rhs) const {
		return heights() == rhs.heights() && widths() == rhs.widths();
	}
private:
	size_t m_height, m_width;
};

class DataBlobShape: public Size {
public:
	DataBlobShape(size_t nums, size_t channels, size_t heights, size_t widths):
		Size(heights, widths), m_num(nums), m_channel(channels) { }
	size_t channels() const { return m_channel; }
	size_t nums() const { return m_num; }
	bool operator == (const DataBlobShape &rhs) const {
		return Size::operator==(rhs) && channels() == rhs.channels() && nums() == rhs.nums();
	}
private:
	size_t m_num;
	size_t m_channel;
};

template <typename T>
class DataBlob {
private:
	size_t m_num;
	size_t m_channel;
	size_t m_height;
	size_t m_width;
	size_t m_step;
	std::shared_ptr<T> m_data;
	size_t m_offset;
	T *raw_ptr() { return m_data.get() + m_offset; }
	const T *raw_ptr() const { return m_data.get() + m_offset; }
public:
	DataBlob();
	DataBlob(size_t num, size_t channel, size_t height, size_t width);
	DataBlob(DataBlobShape shape);
	// do not try to manage data by shared_ptr
	DataBlob(size_t num, size_t channel, size_t height, size_t width, T *data);
	DataBlob(DataBlobShape shape, T *data);
	// shallow-copy constructor
	DataBlob(const DataBlob<T> &rhs);
	DataBlob<T> &operator=(const DataBlob<T> &rhs);

	T &at(size_t n, size_t c, size_t h, size_t w);
	const T &at(size_t n, size_t c, size_t h, size_t w) const;

	DataBlob<T> clone() const;

	// read data from src
	void read(const T *src);
	// write data to dst
	void write(T *dst) const;

	const T *ptr(size_t i = 0) const {
		return raw_ptr() + i* inst_n_elem();
	}
	T *ptr(size_t r = 0) {
		return raw_ptr() + r * inst_n_elem();
	}
	size_t nums() const { return m_num; }
	size_t channels() const { return m_channel; }
	size_t heights() const { return m_height; }
	size_t widths() const { return m_width;}
	size_t step() const { return m_step; }
	size_t inst_n_elem() const {return heights() * widths() * channels();}
	size_t total_n_elem() const {return nums() * inst_n_elem();}
	bool equals(const DataBlob<T> &rhs) const;
	bool is_continuous() const;
	Size size() const { return {heights(), widths()}; }
	DataBlobShape shape() const { return {nums(), channels(), heights(), widths()}; }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const DataBlob<T> &m);

// type aliases
using uchar = unsigned char;
using ushort = unsigned short;
using DataBlob8u = DataBlob<uchar>;
using DataBlob32f = DataBlob<float>;

#endif